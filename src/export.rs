//! Video Export System for Particle Studio RS
//!
//! Goals:
//! - Stable FFmpeg piping (no deadlocks on stderr)
//! - Consistent pixel pipeline (RGBA in, container-specific out)
//! - Alpha-capable formats (ProRes 4444 / VP9 WebM / PNG sequence)
//! - Optional audio muxing (so the render is "под музыку")

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use image::{self, ColorType};

/// Export format options
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExportFormat {
    /// H.264 MP4 (no alpha)
    MP4,
    /// ProRes 4444 MOV (with alpha channel)
    MovAlpha,
    /// WebM VP9 (with alpha)
    WebM,
    /// PNG sequence (with alpha)
    PngSequence,
}

impl Default for ExportFormat {
    fn default() -> Self {
        Self::MP4
    }
}

impl ExportFormat {
    pub fn name(&self) -> &'static str {
        match self {
            Self::MP4 => "MP4 (H.264)",
            Self::MovAlpha => "MOV (ProRes 4444, Alpha)",
            Self::WebM => "WebM (VP9, Alpha)",
            Self::PngSequence => "PNG Sequence (RGBA)",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            Self::MP4 => "mp4",
            Self::MovAlpha => "mov",
            Self::WebM => "webm",
            Self::PngSequence => "png",
        }
    }

    pub fn supports_alpha(&self) -> bool {
        matches!(self, Self::MovAlpha | Self::WebM | Self::PngSequence)
    }
}

/// Video exporter state
pub struct VideoExporter {
    pub format: ExportFormat,
    pub output_path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub total_frames: usize,
    pub current_frame: usize,
    pub is_exporting: bool,
    pub is_cancelled: bool,
    pub progress: f32,
    pub error_message: Option<String>,

    /// Optional audio file to mux into containers.
    pub audio_path: Option<PathBuf>,

    /// MP4 quality controls (libx264)
    pub mp4_preset: String,
    pub mp4_crf: u32,
    pub mp4_pix_fmt: String,

    // Pipe to FFmpeg (containers)
    ffmpeg_process: Option<Child>,
    stderr_tail: Option<Arc<Mutex<VecDeque<String>>>>,
    stderr_thread: Option<JoinHandle<()>>,
}

impl Default for VideoExporter {
    fn default() -> Self {
        Self {
            format: ExportFormat::MP4,
            output_path: PathBuf::new(),
            width: 1920,
            height: 1080,
            fps: 30,
            total_frames: 0,
            current_frame: 0,
            is_exporting: false,
            is_cancelled: false,
            progress: 0.0,
            error_message: None,
            audio_path: None,
            mp4_preset: "slow".to_string(),
            mp4_crf: 18,
            // IMPORTANT: yuv420p is best compatibility.
            // If you want razor-sharp chroma edges (closest to preview RGB),
            // try "yuv444p" + profile high444, but some players won't like it.
            mp4_pix_fmt: "yuv420p".to_string(),
            ffmpeg_process: None,
            stderr_tail: None,
            stderr_thread: None,
        }
    }
}

impl VideoExporter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start export process.
    ///
    /// - `audio_path`: optional path to the source music file.
    pub fn start(
        &mut self,
        output_path: PathBuf,
        width: u32,
        height: u32,
        fps: u32,
        total_frames: usize,
        audio_path: Option<PathBuf>,
    ) -> Result<(), String> {
        self.output_path = output_path;
        self.width = width;
        self.height = height;
        self.fps = fps;
        self.total_frames = total_frames;
        self.current_frame = 0;
        self.is_exporting = true;
        self.is_cancelled = false;
        self.progress = 0.0;
        self.error_message = None;
        self.audio_path = audio_path;

        // PNG sequence: ensure folder exists.
        if self.format == ExportFormat::PngSequence {
            if let Some(parent) = self.output_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            return Ok(());
        }

        self.start_ffmpeg()?;
        Ok(())
    }

    fn start_ffmpeg(&mut self) -> Result<(), String> {
        let output_str = self.output_path.to_string_lossy().to_string();

        let mut cmd = Command::new("ffmpeg");

        // Raw RGBA frames in
        cmd.arg("-y")
            .arg("-nostdin")
            .arg("-loglevel")
            .arg("error")
            .args(["-f", "rawvideo"])
            .args(["-pix_fmt", "rgba"])
            .arg("-s")
            .arg(format!("{}x{}", self.width, self.height))
            .arg("-r")
            .arg(self.fps.to_string())
            .args(["-i", "pipe:0"]);

        // Optional audio input (containers only)
        if let Some(ref audio_path) = self.audio_path {
            cmd.arg("-i").arg(audio_path);
            cmd.args(["-map", "0:v:0", "-map", "1:a:0", "-shortest"]);
        }

        match self.format {
            ExportFormat::MP4 => {
                let crf = self.mp4_crf.clamp(0, 51);
                let pix_fmt = self.mp4_pix_fmt.clone();

                cmd.args(["-c:v", "libx264"])
                    .args(["-preset", &self.mp4_preset])
                    .args(["-crf", &crf.to_string()])
                    .args(["-tune", "animation"])
                    .args(["-pix_fmt", &pix_fmt])
                    .args(["-movflags", "+faststart"]);

                // If user forced yuv444p or yuv422p, set matching profile.
                if pix_fmt == "yuv444p" {
                    cmd.args(["-profile:v", "high444"]);
                } else if pix_fmt == "yuv422p" {
                    cmd.args(["-profile:v", "high422"]);
                }

                if self.audio_path.is_some() {
                    cmd.args(["-c:a", "aac", "-b:a", "256k"]);
                }

                cmd.arg(&output_str);
            }

            ExportFormat::MovAlpha => {
                // ProRes 4444 with alpha (10-bit). Editor-friendly.
                cmd.args(["-c:v", "prores_ks"])
                    // IMPORTANT: profile:v 4 = 4444 (alpha)
                    .args(["-profile:v", "4"])
                    .args(["-pix_fmt", "yuva444p10le"])
                    .args(["-vendor", "ap10"])
                    // Higher = higher bitrate/quality. This is a strong default for graphics.
                    .args(["-bits_per_mb", "8000"]);

                if self.audio_path.is_some() {
                    // Lossless/uncompressed audio for editing.
                    cmd.args(["-c:a", "pcm_s16le"]);
                }

                cmd.arg(&output_str);
            }

            ExportFormat::WebM => {
                let crf = self.mp4_crf.clamp(0, 63); // VP9 uses 0..63-ish
                cmd.args(["-c:v", "libvpx-vp9"])
                    .args(["-pix_fmt", "yuva420p"])
                    .args(["-b:v", "0"])
                    .args(["-crf", &crf.to_string()])
                    .args(["-deadline", "good"])
                    .args(["-cpu-used", "0"])
                    .args(["-row-mt", "1"])
                    .args(["-threads", "8"])
                    // Required in many builds to preserve alpha
                    .args(["-auto-alt-ref", "0"]);

                if self.audio_path.is_some() {
                    cmd.args(["-c:a", "libopus", "-b:a", "192k"]);
                }

                cmd.arg(&output_str);
            }

            ExportFormat::PngSequence => {
                // handled outside
            }
        }

        cmd.stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            format!(
                "Failed to start FFmpeg: {}. Make sure FFmpeg is installed and in PATH.",
                e
            )
        })?;

        // Drain stderr to avoid deadlocks; keep a tail for debugging.
        let tail: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::with_capacity(200)));
        let tail_clone = Arc::clone(&tail);
        let stderr_thread = child.stderr.take().map(|stderr| {
            std::thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().flatten() {
                    let mut q = tail_clone.lock().unwrap();
                    if q.len() >= 200 {
                        q.pop_front();
                    }
                    q.push_back(line);
                }
            })
        });

        self.stderr_tail = Some(tail);
        self.stderr_thread = stderr_thread;
        self.ffmpeg_process = Some(child);
        Ok(())
    }

    /// Write a frame to the export.
    ///
    /// Input must be **RGBA8** (sRGB in RGB channels).
    pub fn write_frame(&mut self, rgba_data: &[u8]) -> Result<(), String> {
        if !self.is_exporting || self.is_cancelled {
            return Err("Export not active".to_string());
        }

        if rgba_data.len() != (self.width as usize * self.height as usize * 4) {
            return Err(format!(
                "Invalid frame buffer size: got {}, expected {}",
                rgba_data.len(),
                self.width as usize * self.height as usize * 4
            ));
        }

        if self.format == ExportFormat::PngSequence {
            let frame_path = self
                .output_path
                .with_file_name(format!("frame_{:06}.png", self.current_frame));

            image::save_buffer(
                &frame_path,
                rgba_data,
                self.width,
                self.height,
                ColorType::Rgba8,
            )
            .map_err(|e| e.to_string())?;
        } else {
            let process = self
                .ffmpeg_process
                .as_mut()
                .ok_or_else(|| "FFmpeg process not started".to_string())?;

            let stdin = process
                .stdin
                .as_mut()
                .ok_or_else(|| "Failed to open FFmpeg stdin".to_string())?;

            stdin.write_all(rgba_data).map_err(|e| e.to_string())?;
        }

        self.current_frame += 1;
        self.progress = self.current_frame as f32 / self.total_frames.max(1) as f32;

        if self.current_frame >= self.total_frames {
            self.finish()?;
        }

        Ok(())
    }

    /// Finish export
    pub fn finish(&mut self) -> Result<(), String> {
        self.is_exporting = false;

        if let Some(mut process) = self.ffmpeg_process.take() {
            // Close stdin to signal EOF
            drop(process.stdin.take());

            let status = process
                .wait()
                .map_err(|e| format!("Failed to wait for FFmpeg: {}", e))?;

            // Ensure stderr thread finished and we have a tail.
            if let Some(handle) = self.stderr_thread.take() {
                let _ = handle.join();
            }

            if !status.success() {
                let tail = self
                    .stderr_tail
                    .take()
                    .map(|q| q.lock().unwrap().iter().cloned().collect::<Vec<_>>().join("\n"))
                    .unwrap_or_else(|| "".to_string());

                return Err(format!(
                    "FFmpeg exited with error code: {:?}\n{}",
                    status.code(),
                    tail
                ));
            }
        }

        Ok(())
    }

    /// Cancel export
    #[allow(dead_code)]
    pub fn cancel(&mut self) {
        self.is_cancelled = true;
        self.is_exporting = false;

        if let Some(mut process) = self.ffmpeg_process.take() {
            let _ = process.kill();
        }
    }

    /// Check if FFmpeg is available
    pub fn check_ffmpeg_available() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    }
}

/// Utility: build a PNG sequence path pattern (for UI/help).
#[allow(dead_code)]
pub fn png_sequence_hint_path(base: &Path) -> PathBuf {
    base.with_file_name("frame_%06d.png")
}
