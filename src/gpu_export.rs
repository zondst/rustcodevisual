//! GPU-Accelerated Video Export Pipeline
//! Triple-buffered async architecture for 4K 60fps export at 3-5x realtime
//! Uses wgpu for rendering and NVENC hardware encoding via FFmpeg

use crate::audio::{AudioAnalysis, AudioState};
use crate::config::AppConfig;
use crate::gpu_render::{GpuRenderer, GpuParticle, RenderParams};
use crate::export::ExportFormat;
use crossbeam_channel::{bounded, Sender, Receiver};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::process::{Command, Stdio, Child};
use std::sync::mpsc::Sender as MpscSender;
use std::thread;
use image::{ImageBuffer, Rgba};
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::rect::Rect;

/// Export progress message
pub enum GpuExportMessage {
    Progress { current: usize, total: usize, fps: f32 },
    Completed { path: PathBuf },
    Error(String),
}

/// Frame data for encoding pipeline
struct FrameData {
    frame_index: usize,
    pixels: Vec<u8>,
}

/// Hardware encoder type
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HardwareEncoder {
    /// NVIDIA NVENC (fastest)
    Nvenc,
    /// AMD AMF/VCE
    Amf,
    /// Intel QuickSync
    Qsv,
    /// Software x264 fallback
    Software,
}

impl HardwareEncoder {
    /// Detect best available hardware encoder
    pub fn detect() -> Self {
        // Check NVENC
        if Self::check_encoder("h264_nvenc") {
            return Self::Nvenc;
        }
        // Check AMD AMF
        if Self::check_encoder("h264_amf") {
            return Self::Amf;
        }
        // Check Intel QSV
        if Self::check_encoder("h264_qsv") {
            return Self::Qsv;
        }
        // Fallback to software
        Self::Software
    }

    fn check_encoder(encoder: &str) -> bool {
        Command::new("ffmpeg")
            .args(["-hide_banner", "-encoders"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .map(|output| {
                let stdout = String::from_utf8_lossy(&output.stdout);
                stdout.contains(encoder)
            })
            .unwrap_or(false)
    }

    /// Get FFmpeg encoder name
    pub fn ffmpeg_encoder(&self) -> &'static str {
        match self {
            Self::Nvenc => "h264_nvenc",
            Self::Amf => "h264_amf",
            Self::Qsv => "h264_qsv",
            Self::Software => "libx264",
        }
    }

    /// Get encoder-specific options (quality-aware, GPU-optimized).
    ///
    /// `quality` is treated as CRF/CQ-like value where **lower = higher quality**.
    /// The valid range depends on encoder; we clamp to a safe range and map accordingly.
    ///
    /// NOTE: For hardware encoders (especially NVENC), constant-quality mode still needs a
    /// sensible bitrate ceiling, otherwise you can get **blocky macroblock artifacts**
    /// ("pixelated squares" on particles) even when CQ looks reasonable.
    ///
    /// GPU OPTIMIZATION: Uses faster presets (P4/P5) and single-pass encoding to maximize
    /// GPU utilization. The previous P7+multipass caused GPU to wait on CPU too much.
    pub fn ffmpeg_options(&self, quality: u32, fps: u32, width: u32, height: u32) -> Vec<String> {
        // Most FFmpeg encoders use 0..51-ish quality scales (x264 CRF is commonly 15..30).
        let q = quality.clamp(0, 51);
        // Reasonable GOP: 2 seconds
        let gop = (fps.saturating_mul(2)).max(1);

        // Heuristic: animated high-contrast graphics (particles) need much higher bitrate
        // than typical camera footage to avoid block artifacts.
        // We compute a bitrate target from pixels/sec and a bpp factor derived from quality.
        let pixels_per_sec = width as f64 * height as f64 * fps as f64;
        // Bits-per-pixel heuristic:
        // - Particle visuals are *hard* for H.264 (lots of tiny high-contrast details).
        // - We intentionally use a fairly high bpp to avoid macroblocking.
        let bpp: f64 = match q {
            0..=12 => 1.30,
            13..=16 => 1.10,
            17..=20 => 0.90,
            21..=28 => 0.70,
            _ => 0.55,
        };
        let target_mbps = ((pixels_per_sec * bpp) / 1_000_000.0).ceil() as u32;
        // Keep sane bounds so exports don't explode in size.
        let target_mbps = target_mbps.clamp(20, 400);
        let maxrate_mbps = (target_mbps.saturating_mul(2)).clamp(40, 600);
        let buf_mbps = (target_mbps.saturating_mul(4)).clamp(80, 1200);

        // Choose preset based on resolution for optimal GPU utilization
        // Higher resolutions benefit from faster presets to keep GPU saturated
        let is_high_res = width >= 2560 || height >= 1440;
        let is_4k = width >= 3840 || height >= 2160;

        match self {
            Self::Nvenc => {
                // GPU-OPTIMIZED NVENC settings:
                // - P4/P5 instead of P7 for better GPU utilization (P7 is CPU-bound)
                // - Single-pass instead of multipass (2x faster, GPU stays busy)
                // - Lookahead reduced for lower latency
                let preset = if is_4k { "p4" } else if is_high_res { "p5" } else { "p5" };
                let lookahead = if is_4k { 16 } else { 24 };

                vec![
                    // Faster preset = higher GPU utilization
                    "-preset".into(), preset.into(),
                    "-tune".into(), "hq".into(),

                    // Single-pass for speed (multipass is CPU-heavy)
                    "-multipass".into(), "disabled".into(),

                    // Constant-quality VBR (good balance of quality and GPU usage)
                    "-rc".into(), "vbr".into(),
                    "-cq".into(), q.to_string(),

                    // Bitrate envelope prevents quality drops
                    "-b:v".into(), format!("{}M", target_mbps),
                    "-maxrate".into(), format!("{}M", maxrate_mbps),
                    "-bufsize".into(), format!("{}M", buf_mbps),

                    // Adaptive quantization (GPU-accelerated on NVENC)
                    "-spatial-aq".into(), "1".into(),
                    "-temporal-aq".into(), "1".into(),
                    "-aq-strength".into(), "10".into(),

                    // Reduced lookahead for faster encoding
                    "-rc-lookahead".into(), lookahead.to_string(),

                    // B-frames for compression efficiency (GPU handles these well)
                    "-bf".into(), "3".into(),
                    "-b_ref_mode".into(), "middle".into(),

                    // GOP size
                    "-g".into(), gop.to_string(),

                    // GPU surface count for parallel processing
                    "-surfaces".into(), "32".into(),
                ]
            },
            Self::Amf => vec![
                // AMD VCE/AMF optimized settings
                "-quality".into(), "balanced".into(), // "quality" is slow, "balanced" uses GPU better
                "-rc".into(), "vbr_peak".into(),
                "-qp_i".into(), q.to_string(),
                "-qp_p".into(), (q.saturating_add(2)).to_string(),
                "-g".into(), gop.to_string(),
                // Enable pre-analysis for better quality
                "-preanalysis".into(), "true".into(),
                // Use more reference frames
                "-bf".into(), "3".into(),
            ],
            Self::Qsv => vec![
                // Intel QuickSync optimized settings
                "-preset".into(), "faster".into(), // "medium" is slow, "faster" saturates GPU
                "-global_quality".into(), q.to_string(),
                "-g".into(), gop.to_string(),
                // Enable look-ahead for quality
                "-look_ahead".into(), "1".into(),
                "-look_ahead_depth".into(), "40".into(),
            ],
            Self::Software => vec![
                // Software is slower but often gives best quality per bitrate.
                "-preset".into(), "medium".into(), // "slow" is too CPU-intensive
                "-crf".into(), q.to_string(),
                "-g".into(), gop.to_string(),
                "-keyint_min".into(), gop.to_string(),
                "-sc_threshold".into(), "0".into(),
                // Better quality for animated/high-contrast graphics
                "-tune".into(), "animation".into(),
                // Use all CPU threads
                "-threads".into(), "0".into(),
            ],
        }
    }


    pub fn name(&self) -> &'static str {
        match self {
            Self::Nvenc => "NVIDIA NVENC",
            Self::Amf => "AMD AMF",
            Self::Qsv => "Intel QuickSync",
            Self::Software => "Software (x264)",
        }
    }
}

/// GPU Export configuration
pub struct GpuExportConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub start_time_secs: f32,
    pub duration_secs: f32,
    pub format: ExportFormat,
    pub output_path: PathBuf,
    pub audio_path: Option<PathBuf>,
    pub encoder: HardwareEncoder,
    #[allow(dead_code)]
    pub quality: u32, // CRF/CQ value (lower = higher quality)
}


/// Try to query supported pixel formats for a given FFmpeg encoder.
/// This lets us pick the best chroma format (420 vs 444) when available.
fn query_supported_pixel_formats(encoder: &str) -> Vec<String> {
    // Example: ffmpeg -hide_banner -h encoder=libx264
    let out = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-h")
        .arg(format!("encoder={}", encoder))
        .output();

    let Ok(out) = out else { return Vec::new(); };
    let text = String::from_utf8_lossy(&out.stdout);

    let mut formats: Vec<String> = Vec::new();
    let mut collecting = false;

    for line in text.lines() {
        if let Some(idx) = line.find("Supported pixel formats:") {
            collecting = true;
            let rest = &line[idx + "Supported pixel formats:".len()..];
            formats.extend(rest.split_whitespace().map(|s| s.to_string()));
            continue;
        }

        if collecting {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }

            // Some ffmpeg builds wrap the list on following indented lines.
            // Stop when we hit another "Supported ..." header.
            if trimmed.starts_with("Supported ") && trimmed.contains(':') {
                break;
            }

            formats.extend(trimmed.split_whitespace().map(|s| s.to_string()));
        }
    }

    formats
}

/// Pick the best YUV pixel format for H.264 output.
/// Prefer 4:4:4 if supported to preserve sharp neon edges/colors (preview is RGB).
fn pick_best_h264_pix_fmt(encoder: &str) -> String {
    let supported = query_supported_pixel_formats(encoder);
    // Prefer higher chroma fidelity when available
    for fmt in ["yuv444p", "yuv422p", "yuv420p"].iter() {
        if supported.iter().any(|f| f == fmt) {
            return (*fmt).to_string();
        }
    }
    // Fallback
    "yuv420p".to_string()
}

/// Build a PNG sequence output pattern from a single file path.
///
/// input:  /path/out.png
/// output: /path/out_%06d.png
fn png_sequence_pattern(output_path: &PathBuf) -> PathBuf {
    let parent = output_path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("frame");
    parent.join(format!("{}_%06d.png", stem))
}

/// Check if CUDA hwupload is available for FFmpeg
fn check_cuda_hwupload_available() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-filters"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map(|output| {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout.contains("hwupload_cuda")
        })
        .unwrap_or(false)
}

/// Check if AMD AMF hwupload is available
fn check_amf_hwupload_available() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-filters"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .map(|output| {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout.contains("hwupload_amf") || stdout.contains("hwupload")
        })
        .unwrap_or(false)
}

/// Start FFmpeg process with the selected output format.
/// Uses GPU hardware upload when available for maximum GPU utilization.
fn start_ffmpeg(config: &GpuExportConfig) -> Result<Child, String> {
    let mut cmd = Command::new("ffmpeg");

    // Check for hardware upload capability
    let use_cuda_upload = config.encoder == HardwareEncoder::Nvenc && check_cuda_hwupload_available();
    let use_amf_upload = config.encoder == HardwareEncoder::Amf && check_amf_hwupload_available();

    // Disable interactive mode, keep output minimal (we'll drain stderr separately).
    cmd.arg("-y")
        .arg("-nostdin")
        .arg("-loglevel").arg("error")
        .arg("-vsync").arg("cfr");

    // Initialize hardware device for GPU upload (NVIDIA CUDA)
    if use_cuda_upload {
        cmd.args(["-init_hw_device", "cuda=cu:0"]);
        cmd.args(["-filter_hw_device", "cu"]);
    }

    // Initialize hardware device for AMD
    if use_amf_upload {
        cmd.args(["-init_hw_device", "d3d11va=hw"]);
        cmd.args(["-filter_hw_device", "hw"]);
    }

    // Raw RGBA frames from GPU renderer
    cmd.arg("-f").arg("rawvideo")
        .arg("-pix_fmt").arg("rgba")
        .arg("-s").arg(format!("{}x{}", config.width, config.height))
        .arg("-r").arg(config.fps.to_string())
        .arg("-thread_queue_size").arg("1024") // Larger input buffer
        .arg("-i").arg("pipe:0");

    // Optional audio input (containers only; PNG sequence ignores audio)
    if let Some(ref audio_path) = config.audio_path {
        if config.format != ExportFormat::PngSequence {
            cmd.arg("-i").arg(audio_path);
            // Explicit stream mapping avoids surprises with containers/codecs
            cmd.args(["-map", "0:v:0", "-map", "1:a:0", "-shortest"]);
        }
    }

    match config.format {
        ExportFormat::MP4 => {
            let encoder_name = config.encoder.ffmpeg_encoder();
            let pix_fmt = pick_best_h264_pix_fmt(encoder_name);

            // GPU-accelerated filter chain for hardware upload
            // This transfers frames directly to GPU memory, bypassing CPU
            if use_cuda_upload {
                // CUDA filter chain: convert colorspace on GPU, upload to GPU memory
                let nv_pix_fmt = if pix_fmt == "yuv444p" { "yuv444p" } else { "nv12" };
                cmd.args(["-vf", &format!(
                    "format=rgba,hwupload_cuda,scale_cuda=format={}",
                    nv_pix_fmt
                )]);
            } else if use_amf_upload {
                // AMD hardware upload
                cmd.args(["-vf", "format=nv12,hwupload"]);
            }

            // Video encoder
            cmd.arg("-c:v").arg(encoder_name);

            // Encoder options (quality-aware, GPU-optimized)
            for opt in config
                .encoder
                .ffmpeg_options(config.quality, config.fps, config.width, config.height)
            {
                cmd.arg(opt);
            }

            // If we ended up using 4:4:4/4:2:2 with libx264, set a matching profile.
            if encoder_name == "libx264" {
                if pix_fmt == "yuv444p" {
                    cmd.args(["-profile:v", "high444"]);
                } else if pix_fmt == "yuv422p" {
                    cmd.args(["-profile:v", "high422"]);
                }
            } else if encoder_name == "h264_nvenc" {
                // NVENC requires an explicit profile for 4:4:4 output.
                if pix_fmt == "yuv444p" {
                    cmd.args(["-profile:v", "high444p"]);
                } else {
                    cmd.args(["-profile:v", "high"]);
                }
            }

            // Pixel format (for non-hwupload paths)
            if !use_cuda_upload && !use_amf_upload {
                cmd.arg("-pix_fmt").arg(&pix_fmt);
            }

            // Audio for MP4
            if config.audio_path.is_some() {
                cmd.args(["-c:a", "aac", "-b:a", "256k"]);
            }

            // MP4 fast start
            cmd.args(["-movflags", "+faststart"]);

            // Output file
            cmd.arg(&config.output_path);
        }

        ExportFormat::MovAlpha => {
            // ProRes 4444 with alpha (high quality, editor-friendly).
            cmd.args(["-c:v", "prores_ks"])
                .args(["-profile:v", "4"])           // 4 = 4444
                .args(["-pix_fmt", "yuva444p10le"])
                .args(["-vendor", "ap10"])
                .args(["-bits_per_mb", "8000"]);

            // Audio for MOV (keep it high quality for editing)
            if config.audio_path.is_some() {
                cmd.args(["-c:a", "pcm_s16le"]);
            }

            cmd.arg(&config.output_path);
        }

        ExportFormat::WebM => {
            // VP9 supports alpha in WebM via yuva420p.
            cmd.args(["-c:v", "libvpx-vp9"])
                .args(["-pix_fmt", "yuva420p"])
                .args(["-b:v", "0"])
                .args(["-crf", &config.quality.to_string()])
                .args(["-deadline", "good"])
                .args(["-cpu-used", "0"])
                .args(["-row-mt", "1"])
                .args(["-threads", "8"])
                // Required for VP9 alpha in many builds
                .args(["-auto-alt-ref", "0"]);

            if config.audio_path.is_some() {
                cmd.args(["-c:a", "libopus", "-b:a", "192k"]);
            }

            cmd.arg(&config.output_path);
        }

        ExportFormat::PngSequence => {
            // PNG sequence (with alpha). We ignore audio here.
            let pattern = png_sequence_pattern(&config.output_path);
            // Ensure parent folder exists
            if let Some(parent) = pattern.parent() {
                let _ = std::fs::create_dir_all(parent);
            }

            cmd.args(["-f", "image2"])
                .args(["-start_number", "0"])
                .args(["-c:v", "png"])
                // Lower is faster, higher is smaller files.
                .args(["-compression_level", "3"]);

            cmd.arg(pattern);
        }
    }

    // Pipe setup.
    // We pipe stderr and drain it in a separate thread to prevent buffer hangs,
    // while still being able to report real FFmpeg errors.
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    cmd.spawn().map_err(|e| format!("Failed to start FFmpeg: {}", e))
}


/// Run GPU-accelerated export with triple-buffered pipeline
pub fn run_gpu_export(
    app_config: AppConfig,
    audio_analysis: AudioAnalysis,
    export_config: GpuExportConfig,
    progress_tx: MpscSender<GpuExportMessage>,
) {
    // Spawn export thread
    thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_gpu_export_impl(app_config, audio_analysis, export_config, &progress_tx)
        }));
        
        match result {
            Ok(Ok(())) => {},
            Ok(Err(e)) => {
                let _ = progress_tx.send(GpuExportMessage::Error(e));
            }
            Err(panic_info) => {
                let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    format!("Export thread panicked: {}", s)
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    format!("Export thread panicked: {}", s)
                } else {
                    "Export thread panicked with unknown error".to_string()
                };
                let _ = progress_tx.send(GpuExportMessage::Error(msg));
            }
        }
    });
}


fn run_gpu_export_impl(
    config: AppConfig,
    audio_analysis: AudioAnalysis,
    export_config: GpuExportConfig,
    progress_tx: &MpscSender<GpuExportMessage>,
) -> Result<(), String> {
    use crate::audio::{AdaptiveAudioNormalizer, NormalizedAudio};
    use crate::config::{BlendMode, ParticleMode, ParticleShape};
    use crate::particles::ParticleEngine;

    let width = export_config.width;
    let height = export_config.height;
    let fps = export_config.fps.max(1);
    let dt = 1.0 / fps as f32;

    // Render from the very start of the track
    let total_output_frames = (export_config.duration_secs * fps as f32).ceil().max(1.0) as usize;
    let start_frame = (export_config.start_time_secs.max(0.0) * fps as f32).round() as usize;
    let total_sim_frames = start_frame + total_output_frames;

    // 1) Initialize GPU renderer (rendering only; simulation is done on CPU to match preview 1:1)
    let renderer = GpuRenderer::new(width, height, config.particles.count as u32)
        .map_err(|e| format!("Failed to init GPU renderer: {}", e))?;

    // 2) CPU simulation engines (exactly the same as preview path)
    let mut particles = ParticleEngine::new(width as f32, height as f32);
    let colors = config.get_color_scheme();
    particles.update_palette(&colors);

    let mut audio_state = AudioState::new();

    // Adaptive audio normalization (same logic as preview)
    let mut audio_normalizer = AdaptiveAudioNormalizer::new(config.particles.adaptive_window_secs, fps);
    let mut normalized_audio: Option<NormalizedAudio> = None;

    // Reusable GPU particle upload buffer to avoid per-frame allocations.
    let mut gpu_particles: Vec<GpuParticle> = Vec::with_capacity(config.particles.count.max(1024));

    // 3) Start FFmpeg + writer thread with deep buffering for GPU saturation
    let ffmpeg = start_ffmpeg(&export_config)?;

    // GPU-OPTIMIZED: Use 8-frame buffer instead of 3 for better GPU utilization
    // This allows the render thread to stay ahead of the encoder, keeping GPU busy
    // Higher resolutions benefit from larger buffers
    let buffer_depth = if width >= 3840 { 12 } else if width >= 2560 { 10 } else { 8 };
    let (frame_tx, frame_rx): (Sender<FrameData>, Receiver<FrameData>) = bounded(buffer_depth);

    // Spawn FFmpeg writer thread
    let output_path = export_config.output_path.clone();
    let progress_tx_clone = progress_tx.clone();
    let ffmpeg_handle = thread::spawn(move || -> Result<(), String> {
        let mut ffmpeg = ffmpeg;

        // Drain FFmpeg stderr continuously to avoid pipe buffer deadlocks,
        // while keeping a tail for better error reports.
        let stderr_tail: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::with_capacity(200)));
        let stderr_tail_clone = Arc::clone(&stderr_tail);

        let stderr_handle = ffmpeg.stderr.take().map(|stderr| {
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().flatten() {
                    let mut q = stderr_tail_clone.lock().unwrap();
                    if q.len() >= 200 {
                        q.pop_front();
                    }
                    q.push_back(line);
                }
            })
        });

        let mut last_progress_frame = 0;
        let progress_interval = 15;

        // Process frames - use scope to ensure stdin borrow ends before take()
        {
            let stdin = ffmpeg.stdin.take().ok_or("Failed to open FFmpeg stdin")?;
            // GPU-OPTIMIZED: Use BufWriter with large buffer (16MB) to reduce syscalls
            // This significantly improves throughput for high-resolution exports
            let mut buffered_stdin = std::io::BufWriter::with_capacity(16 * 1024 * 1024, stdin);

            for frame in frame_rx {
                // Write frame to FFmpeg with buffering
                buffered_stdin.write_all(&frame.pixels)
                    .map_err(|e| format!("FFmpeg write error: {}", e))?;

                // Send progress updates (from writer thread for accurate encoding progress)
                if frame.frame_index.saturating_sub(last_progress_frame) >= progress_interval {
                    let _ = progress_tx_clone.send(GpuExportMessage::Progress {
                        current: frame.frame_index,
                        total: total_output_frames,
                        fps: 0.0, // Will be calculated from elapsed time
                    });
                    last_progress_frame = frame.frame_index;
                }
            }

            // Flush remaining buffered data before closing
            buffered_stdin.flush().map_err(|e| format!("FFmpeg flush error: {}", e))?;
            // Drop buffered_stdin to close the underlying pipe and signal EOF to FFmpeg
        }

        // Wait for FFmpeg to finish with timeout
        let wait_start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(60);

        loop {
            match ffmpeg.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        if let Some(h) = stderr_handle { let _ = h.join(); }
                        let tail = {
                            let q = stderr_tail.lock().unwrap();
                            q.iter().cloned().collect::<Vec<_>>().join("\n")
                        };
                        return Err(format!("FFmpeg exited with code: {:?}\n{}", status.code(), tail));
                    }
                    if let Some(h) = stderr_handle { let _ = h.join(); }
                    break;
                }
                Ok(None) => {
                    if wait_start.elapsed() > timeout {
                        let _ = ffmpeg.kill();
                        if let Some(h) = stderr_handle { let _ = h.join(); }
                        let tail = {
                            let q = stderr_tail.lock().unwrap();
                            q.iter().cloned().collect::<Vec<_>>().join("\n")
                        };
                        return Err(format!("FFmpeg encoding timeout\n{}", tail));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                Err(e) => {
                    return Err(format!("Failed to wait for FFmpeg: {}", e));
                }
            }
        }

        Ok(())
    });

    // 4) Render loop
    let start_time = std::time::Instant::now();
    // Precompute background (sRGB->linear) each frame (depends on alpha-export mode)
    let srgb_to_linear = |srgb: u8| -> f32 {
        let c = srgb as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };

    for sim_frame_idx in 0..total_sim_frames {
        // ============================================================
        // AUDIO UPDATE (frame-accurate)
        // ============================================================
        let audio_frame_idx = (sim_frame_idx as f32 * audio_analysis.fps as f32 / fps as f32) as usize;
        if audio_frame_idx < audio_analysis.total_frames {
            let frame = audio_analysis.get_frame(audio_frame_idx);
            audio_state.update_from_frame(&frame, config.audio.smoothing);
        }
        audio_state.update_smoothing(dt, config.audio.smoothing, config.audio.beat_attack, config.audio.beat_decay);

        // Adaptive normalization (same as preview)
        let normalized_ref: Option<&NormalizedAudio> = if config.particles.adaptive_audio_enabled {
            let norm = audio_normalizer.normalize(
                audio_state.smooth_bass,
                audio_state.smooth_mid,
                audio_state.smooth_high,
                config.particles.bass_sensitivity,
                config.particles.mid_sensitivity,
                config.particles.high_sensitivity,
                config.particles.adaptive_strength,
            );
            normalized_audio = Some(norm);
            normalized_audio.as_ref()
        } else {
            None
        };

        // ============================================================
        // PARTICLE SIM (CPU) - matches preview logic exactly
        // ============================================================
        let death_spiral_ref = if config.particles.mode == ParticleMode::DeathSpiral {
            Some(&config.death_spiral)
        } else {
            None
        };

        particles.update(
            &config.particles,
            &config.connections,
            death_spiral_ref,
            &audio_state,
            dt,
            normalized_ref,
        );
        particles.update_trails(&config.trails, dt);

        // Optional pre-roll: simulate up to start_time without rendering/encoding.
        if sim_frame_idx < start_frame {
            continue;
        }
        let out_frame_idx = sim_frame_idx - start_frame;

        // Upload particles to GPU (rendering only)
        gpu_particles.clear();
        gpu_particles.extend(particles.get_particles().iter().map(crate::gpu_render::cpu_particle_to_gpu));
        renderer.upload_particles(&gpu_particles);
        let num_particles = gpu_particles.len() as u32;

        // ============================================================
        // RENDER PARAMS (background + shape + bloom/exposure)
        // ============================================================
        let colors = config.get_color_scheme();

        let (bg_r, bg_g, bg_b, bg_a) = if export_config.format.supports_alpha() {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            (
                srgb_to_linear(colors.background[0]),
                srgb_to_linear(colors.background[1]),
                srgb_to_linear(colors.background[2]),
                1.0,
            )
        };

        let shape_id = match config.particles.shape {
            ParticleShape::Circle => 0.0,
            ParticleShape::Diamond => 1.0,
            ParticleShape::Star => 2.0,
            ParticleShape::Ring => 3.0,
            ParticleShape::Triangle => 4.0,
            ParticleShape::Spark => 5.0,
            ParticleShape::Glow => 6.0,
            ParticleShape::Point => 7.0,
        };

        // Match preview global size pulse used in ParticleEngine::draw().
        // In preview: size *= (1.0 + eff_amp * 0.2)
        let eff_amp: f32 = if config.particles.adaptive_audio_enabled {
            normalized_ref.map(|n| n.intensity).unwrap_or(audio_state.amplitude)
        } else {
            audio_state.amplitude
        }
        .clamp(0.0, 1.0);

        let render_params = RenderParams {
            width: width as f32,
            height: height as f32,
            glow_intensity: config.particles.glow_intensity,
            exposure: config.visual.exposure,
            // IMPORTANT: preview-style glow is mostly handled in the particle shader.
            // Bloom is optional; keep it subtle to avoid "blocky" halos.
            bloom_strength: if config.visual.bloom_enabled {
                config.visual.bloom_intensity
            } else {
                0.0
            },
            shape_id,
            bg_r,
            bg_g,
            bg_b,
            bg_a,
            _padding: [
                eff_amp,
                if config.particles.volumetric_rendering {
                    config.particles.volumetric_steps as f32
                } else {
                    0.0
                },
            ],
        };

        // ============================================================
        // GPU RENDER
        // ============================================================
        let additive_blend = matches!(config.particles.blend_mode, BlendMode::Add);
        renderer.render_particles(
            num_particles,
            &render_params,
            [0.0, 0.0, 0.0, 0.0],
            additive_blend,
        );

        // Overlay (spectrum/waveform meters etc) in CPU - shared by preview/export
        let overlay_pixels = render_overlay_cpu(width, height, &audio_state, &config);
        renderer.upload_overlay(&overlay_pixels);

        // Optional bloom
        if render_params.bloom_strength > 0.001 {
            renderer.run_bloom();
        }
        renderer.tonemap(&render_params);

        // Readback
        let pixels = renderer.read_frame();

        if frame_tx.send(FrameData { frame_index: out_frame_idx, pixels }).is_err() {
            return Err("FFmpeg writer thread closed unexpectedly".to_string());
        }

        // Progress every ~0.5 sec
        if out_frame_idx % (fps as usize / 2).max(1) == 0 {
            let elapsed = start_time.elapsed().as_secs_f32();
            let render_fps = (out_frame_idx + 1) as f32 / elapsed.max(0.001);
            let _ = progress_tx.send(GpuExportMessage::Progress {
                current: out_frame_idx,
                total: total_output_frames,
                fps: render_fps,
            });
        }
    }

    drop(frame_tx);

    let ffmpeg_result = ffmpeg_handle
        .join()
        .map_err(|_| "FFmpeg writer thread panicked".to_string())?;
    ffmpeg_result?;

    // Final progress + completion
    let _ = progress_tx.send(GpuExportMessage::Progress {
        current: total_output_frames,
        total: total_output_frames,
        fps: 0.0,
    });

    let _ = progress_tx.send(GpuExportMessage::Completed { path: output_path });
    Ok(())
}


/// Check if GPU export is available
pub fn is_gpu_export_available() -> bool {
    // Try to create a minimal GPU context
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            ..Default::default()
        });

        instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await.is_some()
    })
}

/// Get GPU info string
pub fn get_gpu_info() -> Option<String> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            ..Default::default()
        });

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await?;

        let info = adapter.get_info();
        Some(format!("{} ({:?})", info.name, info.backend))
    })
}

/// Signed Distance Function for a line segment
/// Returns the distance from point p to the closest point on line segment a-b
fn sdf_line_segment(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let pa = (p.0 - a.0, p.1 - a.1);
    let ba = (b.0 - a.0, b.1 - a.1);
    let ba_len_sq = ba.0 * ba.0 + ba.1 * ba.1;
    if ba_len_sq < 0.0001 {
        return (pa.0 * pa.0 + pa.1 * pa.1).sqrt();
    }
    let h = ((pa.0 * ba.0 + pa.1 * ba.1) / ba_len_sq).clamp(0.0, 1.0);
    let d = (pa.0 - ba.0 * h, pa.1 - ba.1 * h);
    (d.0 * d.0 + d.1 * d.1).sqrt()
}

/// Proper premultiplied alpha blending
fn blend_pixel_premultiplied(image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, x: u32, y: u32, src: Rgba<u8>) {
    let dst = *image.get_pixel(x, y);

    let src_a = src.0[3] as f32 / 255.0;
    let dst_a = dst.0[3] as f32 / 255.0;

    // Premultiplied alpha compositing: out = src + dst * (1 - src_a)
    let out_a = src_a + dst_a * (1.0 - src_a);

    if out_a > 0.001 {
        let out_r = (src.0[0] as f32 * src_a + dst.0[0] as f32 * dst_a * (1.0 - src_a)) / out_a;
        let out_g = (src.0[1] as f32 * src_a + dst.0[1] as f32 * dst_a * (1.0 - src_a)) / out_a;
        let out_b = (src.0[2] as f32 * src_a + dst.0[2] as f32 * dst_a * (1.0 - src_a)) / out_a;

        image.put_pixel(x, y, Rgba([
            out_r.clamp(0.0, 255.0) as u8,
            out_g.clamp(0.0, 255.0) as u8,
            out_b.clamp(0.0, 255.0) as u8,
            (out_a * 255.0).clamp(0.0, 255.0) as u8,
        ]));
    }
}

/// Draw an anti-aliased line with proper SDF-based rendering
fn draw_antialiased_line_correct(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    start: (f32, f32),
    end: (f32, f32),
    color: Rgba<u8>,
    thickness: f32,
) {
    let (x0, y0) = start;
    let (x1, y1) = end;
    let width = image.width() as i32;
    let height = image.height() as i32;

    let half_thick = thickness * 0.5;
    let aa_width = 1.5; // Anti-aliasing width in pixels

    // Bounding box with padding for anti-aliasing
    let min_x = (x0.min(x1) - half_thick - aa_width).floor() as i32;
    let max_x = (x0.max(x1) + half_thick + aa_width).ceil() as i32;
    let min_y = (y0.min(y1) - half_thick - aa_width).floor() as i32;
    let max_y = (y0.max(y1) + half_thick + aa_width).ceil() as i32;

    for py in min_y.max(0)..=max_y.min(height - 1) {
        for px in min_x.max(0)..=max_x.min(width - 1) {
            let p = (px as f32 + 0.5, py as f32 + 0.5);

            // Signed distance to line segment
            let dist = sdf_line_segment(p, start, end);

            // Distance from edge of line
            let edge_dist = dist - half_thick;

            // Smooth anti-aliased transition using smoothstep
            let alpha = if edge_dist < -aa_width {
                1.0
            } else if edge_dist > aa_width {
                0.0
            } else {
                let t = (edge_dist + aa_width) / (2.0 * aa_width);
                1.0 - t * t * (3.0 - 2.0 * t) // smoothstep
            };

            if alpha > 0.001 {
                let final_alpha = (color.0[3] as f32 * alpha).clamp(0.0, 255.0) as u8;
                blend_pixel_premultiplied(
                    image,
                    px as u32,
                    py as u32,
                    Rgba([color.0[0], color.0[1], color.0[2], final_alpha]),
                );
            }
        }
    }
}

/// Draw a line with simplified glow effect (2 layers instead of 5 to reduce artifacts)
fn draw_line_with_glow(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    start: (f32, f32),
    end: (f32, f32),
    color: [u8; 3],
    thickness: f32,
    glow_alpha: u8,
) {
    // 1. Outer glow layer (single, wider)
    let glow_thickness = thickness * 3.0;
    let outer_alpha = (glow_alpha as f32 * 0.3).clamp(0.0, 255.0) as u8;
    draw_antialiased_line_correct(
        image,
        start,
        end,
        Rgba([color[0], color[1], color[2], outer_alpha]),
        glow_thickness,
    );

    // 2. Main line (full opacity)
    draw_antialiased_line_correct(
        image,
        start,
        end,
        Rgba([color[0], color[1], color[2], 255]),
        thickness,
    );
}

/// Legacy function kept for compatibility - redirects to new implementation
fn draw_antialiased_line_with_thickness(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    start: (f32, f32),
    end: (f32, f32),
    color: Rgba<u8>,
    thickness: f32,
) {
    draw_antialiased_line_correct(image, start, end, color, thickness);
}

/// Draw a circle outline with optional glow
fn draw_circle_outline(
    image: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    center: (f32, f32),
    radius: f32,
    color: [u8; 3],
    alpha: u8,
    thickness: f32,
) {
    let segments = (radius * 0.5).max(64.0) as usize;
    let angle_step = std::f32::consts::TAU / segments as f32;

    for i in 0..segments {
        let angle1 = i as f32 * angle_step;
        let angle2 = (i + 1) as f32 * angle_step;

        let x1 = center.0 + angle1.cos() * radius;
        let y1 = center.1 + angle1.sin() * radius;
        let x2 = center.0 + angle2.cos() * radius;
        let y2 = center.1 + angle2.sin() * radius;

        let rgba = Rgba([color[0], color[1], color[2], alpha]);
        draw_antialiased_line_with_thickness(image, (x1, y1), (x2, y2), rgba, thickness);
    }
}

/// Software renderer for visual overlays (Waveform/Spectrum)
pub(crate) fn render_overlay_cpu(width: u32, height: u32, audio: &AudioState, config: &AppConfig) -> Vec<u8> {
    let mut image = ImageBuffer::from_pixel(width, height, Rgba([0u8, 0, 0, 0]));
    let colors = config.get_color_scheme();

    // Draw Waveform (enhanced for preview quality match)
    if config.waveform.enabled {
        let waveform_color = colors.waveform;
        // Boost thickness to match preview quality (minimum 3.0 for visibility)
        let thickness = config.waveform.thickness.max(3.0) * 1.5;
        // Enhanced glow alpha for better visibility
        let glow_alpha = ((audio.amplitude * 180.0).min(220.0) + 80.0).min(255.0) as u8;

        match config.waveform.style {
            crate::config::WaveformStyle::Circle => {
                let center_x = width as f32 * config.waveform.position_x;
                let center_y = height as f32 * config.waveform.position_y;
                let base_radius = width.min(height) as f32 * config.waveform.circular_radius;
                let amplitude = config.waveform.amplitude * (1.0 + audio.amplitude * 0.5) * 0.5;

                // Draw inner reference circle (like preview)
                let inner_radius = base_radius * 0.8;
                draw_circle_outline(
                    &mut image,
                    (center_x, center_y),
                    inner_radius,
                    waveform_color,
                    50,
                    1.0,
                );

                // Draw waveform circle with glow
                let samples = audio.waveform.len();
                let angle_step = std::f32::consts::TAU / samples as f32;

                let mut prev_x = 0.0;
                let mut prev_y = 0.0;
                let mut first_x = 0.0;
                let mut first_y = 0.0;

                for (i, &value) in audio.waveform.iter().enumerate() {
                    let angle = i as f32 * angle_step - std::f32::consts::FRAC_PI_2;
                    let radius = base_radius + value * amplitude;
                    let x = center_x + angle.cos() * radius;
                    let y = center_y + angle.sin() * radius;

                    if i == 0 {
                        first_x = x;
                        first_y = y;
                    } else {
                        draw_line_with_glow(&mut image, (prev_x, prev_y), (x, y), waveform_color, thickness, glow_alpha);
                    }
                    prev_x = x;
                    prev_y = y;
                }

                // Close the loop
                draw_line_with_glow(&mut image, (prev_x, prev_y), (first_x, first_y), waveform_color, thickness, glow_alpha);
            },
            crate::config::WaveformStyle::Mirror => {
                let center_y = height as f32 * config.waveform.position_y;
                let amplitude = config.waveform.amplitude * (1.0 + audio.amplitude * 0.5);

                let mut prev_x = 0.0;
                let mut prev_upper_y = center_y;
                let mut prev_lower_y = center_y;

                for (i, &value) in audio.waveform.iter().enumerate() {
                    let x = (i as f32 / audio.waveform.len() as f32) * width as f32;
                    let offset = value.abs() * amplitude;
                    let upper_y = center_y - offset;
                    let lower_y = center_y + offset;

                    if i > 0 {
                        draw_line_with_glow(&mut image, (prev_x, prev_upper_y), (x, upper_y), waveform_color, thickness, glow_alpha);
                        draw_line_with_glow(&mut image, (prev_x, prev_lower_y), (x, lower_y), waveform_color, thickness, glow_alpha);
                    }
                    prev_x = x;
                    prev_upper_y = upper_y;
                    prev_lower_y = lower_y;
                }

                // Draw center line
                let center_color = Rgba([waveform_color[0], waveform_color[1], waveform_color[2], 100]);
                draw_antialiased_line_with_thickness(&mut image, (0.0, center_y), (width as f32, center_y), center_color, 1.0);
            },
            _ => {
                // Line style
                let center_y = height as f32 * config.waveform.position_y;
                let amplitude = config.waveform.amplitude * (1.0 + audio.amplitude * 0.5);

                let mut prev_x = 0.0;
                let mut prev_y = center_y;

                for (i, &sample) in audio.waveform.iter().enumerate() {
                    let x = (i as f32 / audio.waveform.len() as f32) * width as f32;
                    let y = center_y - sample * amplitude;

                    if i > 0 {
                        draw_line_with_glow(&mut image, (prev_x, prev_y), (x, y), waveform_color, thickness, glow_alpha);
                    }
                    prev_x = x;
                    prev_y = y;
                }
            }
        }
    }

    // Draw Spectrum bars with gradient and glow
    if config.spectrum.enabled {
        let bar_count = config.spectrum.bar_count.min(audio.spectrum.len());
        let total_bar_width = width as f32 / bar_count as f32;
        let bar_width = total_bar_width * config.spectrum.bar_width;
        let bar_spacing = total_bar_width * config.spectrum.bar_spacing / 2.0;

        let color_low = colors.spectrum_low;
        let color_high = colors.spectrum_high;

        for i in 0..bar_count {
            let spectrum_idx = i * audio.spectrum.len() / bar_count;
            let val = audio.spectrum.get(spectrum_idx).copied().unwrap_or(0.0);
            let boosted_val = (val * (1.0 + audio.beat * 0.5)).min(1.0);
            let bar_height = boosted_val * config.spectrum.bar_height_scale * height as f32 * 0.5;

            let x = (i as f32 * total_bar_width + bar_spacing) as i32;
            let y = (height as f32 - bar_height) as i32;

            let bw = bar_width as u32;
            let bh = bar_height as u32;

            // Gradient color based on frequency
            let t = i as f32 / bar_count as f32;
            let bar_color = [
                ((1.0 - t) * color_low[0] as f32 + t * color_high[0] as f32) as u8,
                ((1.0 - t) * color_low[1] as f32 + t * color_high[1] as f32) as u8,
                ((1.0 - t) * color_low[2] as f32 + t * color_high[2] as f32) as u8,
            ];

            if bw > 0 && bh > 0 {
                // Draw outer glow (largest, most transparent) - enhanced for preview match
                let outer_glow = Rgba([bar_color[0], bar_color[1], bar_color[2], (boosted_val * 40.0) as u8]);
                let outer_rect = Rect::at((x - 4).max(0), (y - 8).max(0))
                    .of_size((bw + 8).min(width - x.max(0) as u32), (bh + 16).min(height));
                draw_filled_rect_mut(&mut image, outer_rect, outer_glow);

                // Draw mid glow
                let mid_glow = Rgba([bar_color[0], bar_color[1], bar_color[2], (boosted_val * 80.0) as u8]);
                let mid_rect = Rect::at((x - 2).max(0), (y - 4).max(0))
                    .of_size((bw + 4).min(width - x.max(0) as u32), (bh + 8).min(height));
                draw_filled_rect_mut(&mut image, mid_rect, mid_glow);

                // Draw inner glow
                let inner_glow = Rgba([bar_color[0], bar_color[1], bar_color[2], (boosted_val * 150.0).min(200.0) as u8]);
                let inner_rect = Rect::at((x - 1).max(0), (y - 2).max(0))
                    .of_size((bw + 2).min(width - x.max(0) as u32), (bh + 4).min(height));
                draw_filled_rect_mut(&mut image, inner_rect, inner_glow);

                // Draw main bar (full opacity)
                let main_color = Rgba([bar_color[0], bar_color[1], bar_color[2], 255]);
                let bar_rect = Rect::at(x, y).of_size(bw, bh);
                draw_filled_rect_mut(&mut image, bar_rect, main_color);

                // Draw bright top cap
                let cap_height = (2).min(bh as i32);
                if cap_height > 0 {
                    let cap_color = Rgba([
                        (bar_color[0] as u16 + 30).min(255) as u8,
                        (bar_color[1] as u16 + 30).min(255) as u8,
                        (bar_color[2] as u16 + 30).min(255) as u8,
                        255
                    ]);
                    let cap_rect = Rect::at(x, y).of_size(bw, cap_height as u32);
                    draw_filled_rect_mut(&mut image, cap_rect, cap_color);
                }
            }
        }
    }

    image.into_raw()
}
