//! Video Export System for Particle Studio RS
//! Supports MP4, MOV with alpha channel, and PNG sequence export

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::io::Write;
use image::{RgbaImage, ImageBuffer, Rgba};

/// Export format options
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExportFormat {
    /// H.264 MP4 (no alpha)
    MP4,
    /// ProRes 4444 MOV (with alpha channel)
    MOV_Alpha,
    /// WebM VP9 (with alpha)
    WebM,
    /// PNG sequence
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
            Self::MOV_Alpha => "MOV (Alpha)",
            Self::WebM => "WebM (VP9)",
            Self::PngSequence => "PNG Sequence",
        }
    }
    
    pub fn extension(&self) -> &'static str {
        match self {
            Self::MP4 => "mp4",
            Self::MOV_Alpha => "mov",
            Self::WebM => "webm",
            Self::PngSequence => "png",
        }
    }
    
    pub fn supports_alpha(&self) -> bool {
        matches!(self, Self::MOV_Alpha | Self::WebM | Self::PngSequence)
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
    
    // Frame buffer for PNG sequence or pipe to FFmpeg
    frame_buffer: Vec<u8>,
    ffmpeg_process: Option<std::process::Child>,
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
            frame_buffer: Vec::new(),
            ffmpeg_process: None,
        }
    }
}

impl VideoExporter {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Start export process
    pub fn start(&mut self, output_path: PathBuf, width: u32, height: u32, fps: u32, total_frames: usize) -> Result<(), String> {
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
        
        // For PNG sequence, just create directory
        if self.format == ExportFormat::PngSequence {
            if let Some(parent) = self.output_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            return Ok(());
        }
        
        // Start FFmpeg process
        self.start_ffmpeg()?;
        
        Ok(())
    }
    
    fn start_ffmpeg(&mut self) -> Result<(), String> {
        let output_str = self.output_path.to_string_lossy().to_string();
        
        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y") // Overwrite
           .arg("-f").arg("rawvideo")
           .arg("-vcodec").arg("rawvideo")
           .arg("-pix_fmt").arg(if self.format.supports_alpha() { "rgba" } else { "rgb24" })
           .arg("-s").arg(format!("{}x{}", self.width, self.height))
           .arg("-r").arg(self.fps.to_string())
           .arg("-i").arg("-"); // Read from stdin
        
        // Output codec settings
        match self.format {
            ExportFormat::MP4 => {
                cmd.arg("-c:v").arg("libx264")
                   .arg("-preset").arg("medium")
                   .arg("-crf").arg("18")
                   .arg("-pix_fmt").arg("yuv420p");
            }
            ExportFormat::MOV_Alpha => {
                cmd.arg("-c:v").arg("prores_ks")
                   .arg("-profile:v").arg("4444")
                   .arg("-pix_fmt").arg("yuva444p10le");
            }
            ExportFormat::WebM => {
                cmd.arg("-c:v").arg("libvpx-vp9")
                   .arg("-crf").arg("20")
                   .arg("-b:v").arg("0")
                   .arg("-pix_fmt").arg("yuva420p");
            }
            ExportFormat::PngSequence => {
                // Handled separately
                return Ok(());
            }
        }
        
        cmd.arg(&output_str)
           .stdin(Stdio::piped())
           .stdout(Stdio::null())
           .stderr(Stdio::null());
        
        match cmd.spawn() {
            Ok(child) => {
                self.ffmpeg_process = Some(child);
                Ok(())
            }
            Err(e) => {
                Err(format!("Failed to start FFmpeg: {}. Make sure FFmpeg is installed.", e))
            }
        }
    }
    
    /// Write a frame to the export
    pub fn write_frame(&mut self, rgba_data: &[u8]) -> Result<(), String> {
        if !self.is_exporting || self.is_cancelled {
            return Err("Export not active".to_string());
        }
        
        if self.format == ExportFormat::PngSequence {
            // Save as PNG file
            let frame_path = self.output_path.with_file_name(
                format!("frame_{:06}.png", self.current_frame)
            );
            
            let img: RgbaImage = ImageBuffer::from_raw(
                self.width, self.height, rgba_data.to_vec()
            ).ok_or("Failed to create image")?;
            
            img.save(&frame_path).map_err(|e| e.to_string())?;
        } else {
            // Write to FFmpeg stdin
            if let Some(ref mut process) = self.ffmpeg_process {
                if let Some(ref mut stdin) = process.stdin {
                    // Convert RGBA to RGB if needed for MP4
                    let data = if self.format.supports_alpha() {
                        rgba_data.to_vec()
                    } else {
                        // Convert RGBA to RGB
                        rgba_data.chunks(4)
                            .flat_map(|chunk| [chunk[0], chunk[1], chunk[2]])
                            .collect()
                    };
                    
                    stdin.write_all(&data).map_err(|e| e.to_string())?;
                }
            }
        }
        
        self.current_frame += 1;
        self.progress = self.current_frame as f32 / self.total_frames as f32;
        
        // Check if done
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
            
            // Wait for FFmpeg to finish
            match process.wait() {
                Ok(status) if !status.success() => {
                    return Err(format!("FFmpeg exited with error: {:?}", status.code()));
                }
                Err(e) => {
                    return Err(format!("Failed to wait for FFmpeg: {}", e));
                }
                _ => {}
            }
        }
        
        Ok(())
    }
    
    /// Cancel export
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
