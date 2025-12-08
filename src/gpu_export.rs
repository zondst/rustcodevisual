//! GPU-Accelerated Video Export Pipeline
//! Triple-buffered async architecture for 4K 60fps export at 3-5x realtime
//! Uses wgpu for rendering and NVENC hardware encoding via FFmpeg

use crate::audio::{AudioAnalysis, AudioState};
use crate::config::AppConfig;
use crate::gpu_render::{GpuRenderer, GpuParticle, RenderParams, SimParams};
use crate::particles::ParticleEngine;
use crossbeam_channel::{bounded, Sender, Receiver};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio, Child};
use std::sync::mpsc::Sender as MpscSender;
use std::thread;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use image::{ImageBuffer, Rgba};
use imageproc::drawing::{draw_line_segment_mut, draw_filled_rect_mut};
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

    /// Get encoder-specific options
    pub fn ffmpeg_options(&self) -> Vec<&'static str> {
        match self {
            Self::Nvenc => vec![
                "-preset", "p6",
                "-tune", "hq",
                "-rc", "vbr",
                "-cq", "19",
                "-b:v", "0",
                "-maxrate", "80M",
                "-bufsize", "160M",
                "-temporal-aq", "1",
                "-spatial-aq", "1",
                "-rc-lookahead", "20",
                "-bf", "3",
                "-g", "120",
            ],
            Self::Amf => vec![
                "-quality", "quality",
                "-rc", "vbr_peak",
                "-qp_i", "19",
                "-qp_p", "21",
            ],
            Self::Qsv => vec![
                "-preset", "medium",
                "-global_quality", "20",
            ],
            Self::Software => vec![
                "-preset", "medium",
                "-crf", "18",
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
    pub duration_secs: f32,
    pub output_path: PathBuf,
    pub audio_path: Option<PathBuf>,
    pub encoder: HardwareEncoder,
    #[allow(dead_code)]
    pub quality: u32, // CRF/CQ value (lower = higher quality)
}

/// Start FFmpeg process with hardware encoding
fn start_ffmpeg(config: &GpuExportConfig) -> Result<Child, String> {
    let mut cmd = Command::new("ffmpeg");

    // CRITICAL: Disable interactive mode and minimize stderr output
    // This prevents the 99% hang issue caused by stderr buffer filling up
    cmd.arg("-y")
       .arg("-nostdin")           // Disable interactive mode
       .arg("-loglevel").arg("error")  // Only show errors, reduce stderr output
       .arg("-vsync").arg("cfr")
       .arg("-f").arg("rawvideo")
       .arg("-pix_fmt").arg("rgba")
       .arg("-s").arg(format!("{}x{}", config.width, config.height))
       .arg("-r").arg(config.fps.to_string())
       .arg("-i").arg("pipe:0");

    // Add audio input if available
    if let Some(ref audio_path) = config.audio_path {
        cmd.arg("-i").arg(audio_path);
    }

    // Video filter for pixel format conversion
    cmd.arg("-vf").arg("format=yuv420p");

    // Encoder selection
    cmd.arg("-c:v").arg(config.encoder.ffmpeg_encoder());

    // Encoder-specific options
    for opt in config.encoder.ffmpeg_options() {
        cmd.arg(opt);
    }

    // Audio encoding (if audio input exists)
    if config.audio_path.is_some() {
        cmd.arg("-c:a").arg("aac")
           .arg("-b:a").arg("256k");
    }

    // Output optimizations
    cmd.arg("-movflags").arg("+faststart");

    // Output file
    cmd.arg(&config.output_path);

    // Pipe setup - redirect stderr to null to prevent buffer fill hang
    cmd.stdin(Stdio::piped())
       .stdout(Stdio::null())
       .stderr(Stdio::null());  // FIXED: Redirect stderr to null instead of piped

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
    let width = export_config.width;
    let height = export_config.height;
    let fps = export_config.fps;
    let total_frames = (export_config.duration_secs * fps as f32) as usize;

    // 1. Initialize Renderer
    let mut renderer = GpuRenderer::new(width, height, config.particles.count as u32)
        .map_err(|e| format!("Failed to init GPU renderer: {}", e))?;

    // 2. Generate Initial Particles - matching preview quality with audio-reactive initialization
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut gpu_particles = Vec::with_capacity(config.particles.count);
    let colors = config.get_color_scheme();

    // Size factor matching preview - shader now handles volumetric rendering correctly
    // Preview's draw_volumetric_particle uses radius from size*1.5 (outer) to size*0.1 (inner)
    // Shader quad size is controlled by glow_mult in vertex shader (3.0 + glow_intensity * 1.5)
    let size_factor = 1.0;  // No additional boost needed - shader handles this

    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;

    for i in 0..config.particles.count {
        // Initial position - spawn from center area like preview's spawn_audio_particle
        let angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let radius = rng.gen_range(10.0..config.particles.spawn_radius);
        let pos = [cx + angle.cos() * radius, cy + angle.sin() * radius];

        // Small initial velocity, outward from center
        let speed = 0.1 + rng.gen::<f32>() * 0.2;
        let vel = [angle.cos() * speed, angle.sin() * speed];

        // Assign random color from scheme
        let color_idx = rng.gen_range(0..colors.particles.len());
        let p_color = colors.particles[color_idx];
        let color_val = [
            p_color[0] as f32 / 255.0,
            p_color[1] as f32 / 255.0,
            p_color[2] as f32 / 255.0,
            1.0,
        ];

        // Calculate particle size with variation, matching preview exactly
        let base_size = config.particles.min_size + rng.gen::<f32>() * (config.particles.max_size - config.particles.min_size);
        let size_var = 1.0 + (rng.gen::<f32>() - 0.5) * config.particles.size_variation;
        let final_size = base_size * size_var * size_factor;

        // CRITICAL: Match preview's particle lifecycle
        // Preview uses life: 2-5 seconds, audio_alpha: 0.1 (starts low, ramps up)
        // Stagger initial life so particles don't all die at once
        let initial_life = if config.particles.audio_reactive_spawn {
            // For audio-reactive mode: short life like preview, staggered
            0.5 + rng.gen::<f32>() * 2.0 + (i as f32 / config.particles.count as f32) * 2.0
        } else {
            // For non-reactive mode: longer life
            2.0 + rng.gen::<f32>() * 3.0
        };

        // CRITICAL: Start with low alpha like preview (0.1)
        // The shader will ramp this up based on audio
        let initial_alpha = if config.particles.audio_reactive_spawn {
            0.1 // Matches preview's spawn_audio_particle
        } else {
            1.0 // Non-reactive mode: fully visible
        };

        gpu_particles.push(GpuParticle {
            position: pos,
            velocity: vel,
            color: color_val,
            size: final_size,
            life: initial_life,
            max_life: 5.0, // Matches preview's max_life
            audio_alpha: initial_alpha,
            audio_size: 0.5,
            brightness: 1.0,
            _padding: [0.0; 2],
        });
    }

    renderer.upload_particles(&gpu_particles);

    // 3. Start FFmpeg with triple-buffered async pipeline
    let ffmpeg = start_ffmpeg(&export_config)?;

    // Create bounded channel for triple-buffering (3 frames in flight)
    let (frame_tx, frame_rx): (Sender<FrameData>, Receiver<FrameData>) = bounded(3);

    // Spawn FFmpeg writer thread
    let output_path = export_config.output_path.clone();
    let progress_tx_clone = progress_tx.clone();
    let ffmpeg_handle = thread::spawn(move || -> Result<(), String> {
        let mut ffmpeg = ffmpeg;
        let stdin = ffmpeg.stdin.as_mut().ok_or("Failed to open FFmpeg stdin")?;

        let mut last_progress_frame = 0;
        let progress_interval = 15;

        for frame in frame_rx {
            // Write frame to FFmpeg
            stdin.write_all(&frame.pixels)
                .map_err(|e| format!("FFmpeg write error: {}", e))?;

            // Send progress updates (from writer thread for accurate encoding progress)
            if frame.frame_index - last_progress_frame >= progress_interval {
                let _ = progress_tx_clone.send(GpuExportMessage::Progress {
                    current: frame.frame_index,
                    total: total_frames,
                    fps: 0.0, // Will be calculated from elapsed time
                });
                last_progress_frame = frame.frame_index;
            }
        }

        // Close stdin to signal EOF
        drop(stdin);

        // Wait for FFmpeg to finish with timeout
        let wait_start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(60);

        loop {
            match ffmpeg.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        return Err(format!("FFmpeg exited with code: {:?}", status.code()));
                    }
                    break;
                }
                Ok(None) => {
                    if wait_start.elapsed() > timeout {
                        let _ = ffmpeg.kill();
                        return Err("FFmpeg encoding timeout".to_string());
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

    // 4. Audio State
    let mut audio_state = AudioState::new();

    // 5. Render Loop (async - doesn't wait for FFmpeg writes)
    let dt = 1.0 / fps as f32;
    let mut time = 0.0;
    let start_time = std::time::Instant::now();

    for frame_idx in 0..total_frames {
        // Audio Update
        let audio_frame_idx = (frame_idx as f32 * audio_analysis.fps as f32 / fps as f32) as usize;
        if audio_frame_idx < audio_analysis.total_frames {
            let frame = audio_analysis.get_frame(audio_frame_idx);
            audio_state.update_from_frame(&frame, config.audio.smoothing);
        }

        // Apply adaptive audio normalization if enabled (matching preview behavior)
        let (normalized_amplitude, normalized_bass, normalized_mid, normalized_high) =
            if config.particles.adaptive_audio_enabled {
                // Simple adaptive normalization: boost quiet sections, compress loud sections
                // This matches the preview's NormalizedAudio behavior
                let base_amp = audio_state.amplitude;
                let adaptive = config.particles.adaptive_strength;

                // Apply frequency-weighted sensitivity (matching preview)
                let bass = (audio_state.bass * config.particles.bass_sensitivity).min(1.5);
                let mid = (audio_state.mid * config.particles.mid_sensitivity).min(1.5);
                let high = (audio_state.high * config.particles.high_sensitivity).min(1.5);

                // Compute weighted intensity
                let intensity = (bass * 0.4 + mid * 0.35 + high * 0.25).clamp(0.0, 1.5);

                // Blend between raw and normalized based on adaptive_strength
                let final_amp = base_amp * (1.0 - adaptive * 0.5) + intensity * adaptive * 0.5;

                (final_amp.clamp(0.0, 1.5), bass, mid, high)
            } else {
                (audio_state.amplitude, audio_state.bass, audio_state.mid, audio_state.high)
            };

        // Sim Params - use actual config values for proper physics matching preview
        let sim_params = SimParams {
            delta_time: dt,
            time,
            width: width as f32,
            height: height as f32,
            audio_amplitude: normalized_amplitude,
            audio_bass: normalized_bass,
            audio_mid: normalized_mid,
            audio_high: normalized_high,
            audio_beat: audio_state.beat,
            beat_burst_strength: config.particles.beat_burst_strength,
            damping: config.particles.damping,
            speed: config.particles.speed,
            num_particles: config.particles.count as u32,
            has_audio: 1,
            // Audio-reactive parameters matching preview
            fade_attack_speed: config.particles.fade_attack_speed,
            fade_release_speed: config.particles.fade_release_speed,
            audio_spawn_threshold: config.particles.audio_spawn_threshold,
            audio_reactive_spawn: if config.particles.audio_reactive_spawn { 1 } else { 0 },
            spawn_radius: config.particles.spawn_radius,
            gravity: config.particles.gravity,
            _padding: [0.0; 2],
        };

        // Render Params - matched to preview quality exactly
        let render_params = RenderParams {
            width: width as f32,
            height: height as f32,
            glow_intensity: config.particles.glow_intensity, // Use exact config value - shader handles volumetric rendering
            exposure: 1.0,
            bloom_strength: config.visual.bloom_intensity,
            shape_id: match config.particles.shape {
                crate::config::ParticleShape::Circle => 0.0,
                crate::config::ParticleShape::Diamond => 1.0,
                crate::config::ParticleShape::Star => 2.0,
                _ => 0.0,
            },
            _padding: [0.0; 2],
        };

        // GPU Execution
        renderer.upload_spectrum(&audio_state.spectrum);
        renderer.simulate_particles(&sim_params);

        // Get background color - convert sRGB to linear for correct rendering
        // The output texture (Rgba8UnormSrgb) will convert linear back to sRGB
        let colors = config.get_color_scheme();
        let srgb_to_linear = |srgb: u8| -> f32 {
            let s = srgb as f32 / 255.0;
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        let bg = [
            srgb_to_linear(colors.background[0]),
            srgb_to_linear(colors.background[1]),
            srgb_to_linear(colors.background[2]),
            1.0
        ];

        renderer.render_particles(config.particles.count as u32, &render_params, bg);

        // Render and upload overlay
        let overlay_pixels = render_overlay_cpu(width, height, &audio_state, &config);
        renderer.upload_overlay(&overlay_pixels);

        renderer.tonemap(&render_params);

        // Readback frame from GPU
        let pixels = renderer.read_frame();

        // Send to FFmpeg writer thread (non-blocking with triple-buffer)
        // This allows GPU to continue rendering while FFmpeg writes
        if frame_tx.send(FrameData { frame_index: frame_idx, pixels }).is_err() {
            return Err("FFmpeg writer thread closed unexpectedly".to_string());
        }

        time += dt;

        // Update progress with render FPS (separate from encoding progress)
        if frame_idx % 30 == 0 {
            let elapsed = start_time.elapsed().as_secs_f32();
            let render_fps = frame_idx as f32 / elapsed.max(0.001);
            let _ = progress_tx.send(GpuExportMessage::Progress {
                current: frame_idx,
                total: total_frames,
                fps: render_fps,
            });
        }
    }

    // Close frame channel to signal completion
    drop(frame_tx);

    // Wait for FFmpeg writer thread to finish
    let ffmpeg_result = ffmpeg_handle.join()
        .map_err(|_| "FFmpeg writer thread panicked".to_string())?;

    // Check for FFmpeg errors
    ffmpeg_result?;

    // Send 100% progress after successful completion
    let _ = progress_tx.send(GpuExportMessage::Progress {
        current: total_frames,
        total: total_frames,
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
fn render_overlay_cpu(width: u32, height: u32, audio: &AudioState, config: &AppConfig) -> Vec<u8> {
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
