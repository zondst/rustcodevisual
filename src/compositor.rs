//! Compositor for Particle Studio RS
//! Combines all visualization layers and handles video export

use image::ImageBuffer;
use std::io::Write;
use std::process::{Child, Command, Stdio};

use crate::audio::{AudioAnalysis, AudioState};
use crate::config::AppConfig;
use crate::particles::ParticleEngine;
use crate::postprocess::{BackgroundRenderer, FrameBuffer, PostProcessor};

/// Frame compositor combining all layers
#[allow(dead_code)]
pub struct Compositor {
    pub width: u32,
    pub height: u32,

    // Frame buffer
    pub buffer: FrameBuffer,

    // Layers
    pub background: BackgroundRenderer,
    pub postprocess: PostProcessor,
}

impl Compositor {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            buffer: ImageBuffer::new(width, height),
            background: BackgroundRenderer::new(width, height),
            postprocess: PostProcessor::new(width, height),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if self.width != width || self.height != height {
            self.width = width;
            self.height = height;
            self.buffer = ImageBuffer::new(width, height);
            self.background.resize(width, height);
            self.postprocess.resize(width, height);
        }
    }

    /// Render a complete frame
    pub fn render_frame(&mut self, config: &AppConfig, audio: &AudioState) {
        let colors = config.get_color_scheme();

        // 1. Render background
        self.background.render(
            &mut self.buffer,
            &colors,
            config.background.animated,
            config.background.intensity,
            audio,
        );

        // 2. Post-processing is applied after particles are rendered to buffer
        // (Particles are rendered via egui, so we apply postprocess to buffer separately if needed)
    }

    /// Update animated elements
    pub fn update(&mut self, dt: f32, audio: &AudioState) {
        self.background.update(dt, audio.amplitude);
    }

    /// Apply post-processing to the buffer
    pub fn apply_postprocess(&mut self, config: &AppConfig, audio: &AudioState) {
        self.postprocess
            .process(&mut self.buffer, &config.visual, audio);
    }

    /// Get buffer as raw bytes (RGB)
    pub fn get_buffer_bytes(&self) -> &[u8] {
        self.buffer.as_raw()
    }

    /// Convert egui texture to buffer for post-processing
    /// This is used when we need to capture the egui render and apply effects
    pub fn copy_from_pixels(&mut self, pixels: &[u8], width: u32, height: u32) {
        if width != self.width || height != self.height {
            self.resize(width, height);
        }

        for (i, pixel) in self.buffer.pixels_mut().enumerate() {
            let offset = i * 3;
            if offset + 2 < pixels.len() {
                pixel[0] = pixels[offset];
                pixel[1] = pixels[offset + 1];
                pixel[2] = pixels[offset + 2];
            }
        }
    }
}

/// Video renderer for exporting to file
#[allow(dead_code)]
pub struct VideoRenderer {
    width: u32,
    height: u32,
    fps: u32,
    output_path: String,
    audio_path: Option<String>,
    ffmpeg_process: Option<Child>,
}

impl VideoRenderer {
    pub fn new(
        width: u32,
        height: u32,
        fps: u32,
        output_path: String,
        audio_path: Option<String>,
    ) -> Self {
        Self {
            width,
            height,
            fps,
            output_path,
            audio_path,
            ffmpeg_process: None,
        }
    }

    /// Start FFmpeg process for encoding
    pub fn start(&mut self, config: &crate::config::ExportConfig) -> anyhow::Result<()> {
        let mut args = vec![
            "-y".to_string(),
            "-f".to_string(),
            "rawvideo".to_string(),
            "-pix_fmt".to_string(),
            "rgb24".to_string(),
            "-s".to_string(),
            format!("{}x{}", self.width, self.height),
            "-r".to_string(),
            self.fps.to_string(),
            "-i".to_string(),
            "-".to_string(), // Read from stdin
        ];

        // Add audio input if available
        if let Some(ref audio_path) = self.audio_path {
            args.extend(vec![
                "-i".to_string(),
                audio_path.clone(),
                "-c:a".to_string(),
                config.codec.clone(),
                "-b:a".to_string(),
                "192k".to_string(),
            ]);
        }

        // Video encoding settings
        args.extend(vec![
            "-c:v".to_string(),
            config.codec.clone(),
            "-preset".to_string(),
            config.preset.clone(),
            "-crf".to_string(),
            config.crf.to_string(),
            "-pix_fmt".to_string(),
            "yuv420p".to_string(),
        ]);

        // Map audio if present
        if self.audio_path.is_some() {
            args.extend(vec![
                "-map".to_string(),
                "0:v".to_string(),
                "-map".to_string(),
                "1:a".to_string(),
                "-shortest".to_string(),
            ]);
        }

        args.push(self.output_path.clone());

        let process = Command::new("ffmpeg")
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;

        self.ffmpeg_process = Some(process);
        Ok(())
    }

    /// Write a frame to FFmpeg
    pub fn write_frame(&mut self, buffer: &FrameBuffer) -> anyhow::Result<()> {
        if let Some(ref mut process) = self.ffmpeg_process {
            if let Some(ref mut stdin) = process.stdin {
                stdin.write_all(buffer.as_raw())?;
            }
        }
        Ok(())
    }

    /// Finish encoding and close FFmpeg
    pub fn finish(&mut self) -> anyhow::Result<()> {
        if let Some(mut process) = self.ffmpeg_process.take() {
            // Close stdin to signal end of input
            drop(process.stdin.take());

            // Wait for process to complete
            process.wait()?;
        }
        Ok(())
    }
}

/// Render an entire visualization to video
#[allow(dead_code)]
pub fn render_to_video(
    config: &AppConfig,
    analysis: &AudioAnalysis,
    output_path: &str,
    progress_callback: Option<Box<dyn Fn(f32) + Send>>,
) -> anyhow::Result<()> {
    let width = config.export.width;
    let height = config.export.height;
    let fps = config.export.fps;

    let mut compositor = Compositor::new(width, height);
    let mut particles = ParticleEngine::new(width as f32, height as f32);
    particles.update_palette(&config.get_color_scheme());

    let mut audio_state = AudioState::new();

    let mut renderer = VideoRenderer::new(
        width,
        height,
        fps,
        output_path.to_string(),
        None, // TODO: audio path
    );

    renderer.start(&config.export)?;

    let total_frames = analysis.total_frames;
    let dt = 1.0 / fps as f32;

    for frame_idx in 0..total_frames {
        // Get audio frame
        let audio_frame = analysis.get_frame(frame_idx);
        audio_state.update_from_frame(&audio_frame, config.audio.smoothing);

        // Update
        compositor.update(dt, &audio_state);
        // Pass Death Spiral config if relevant (Option)
        let death_spiral_config =
            if config.particles.mode == crate::config::ParticleMode::DeathSpiral {
                Some(&config.death_spiral)
            } else {
                None
            };

        particles.update(
            &config.particles,
            &config.connections,
            death_spiral_config,
            &audio_state,
            dt,
            None,
        );

        // Render background
        compositor.render_frame(config, &audio_state);

        // Render particles to buffer
        // Note: In a full implementation, we'd render particles to a texture
        // For now, we'll render a simple version directly to the buffer
        render_particles_to_buffer(
            &particles,
            &mut compositor.buffer,
            &config.particles,
            &audio_state,
        );

        // Apply post-processing
        compositor.apply_postprocess(config, &audio_state);

        // Write frame
        renderer.write_frame(&compositor.buffer)?;

        // Progress callback
        if let Some(ref callback) = progress_callback {
            callback((frame_idx as f32 + 1.0) / total_frames as f32);
        }
    }

    renderer.finish()?;
    Ok(())
}

/// Render particles directly to image buffer
fn render_particles_to_buffer(
    engine: &ParticleEngine,
    buffer: &mut FrameBuffer,
    config: &crate::config::ParticleConfig,
    audio: &AudioState,
) {
    if !config.enabled {
        return;
    }

    let width = buffer.width();
    let height = buffer.height();

    for p in engine.particles.iter() {
        let life_alpha = (p.life / p.max_life).clamp(0.0, 1.0);
        let alpha = (life_alpha * p.brightness).min(1.0);

        if alpha < 0.02 {
            continue;
        }

        let size = (p.size * (1.0 + audio.amplitude * 0.3)) as i32;
        let x = p.pos.x as i32;
        let y = p.pos.y as i32;

        // Draw filled circle (simple rasterization)
        for dy in -size..=size {
            for dx in -size..=size {
                if dx * dx + dy * dy <= size * size {
                    let px = x + dx;
                    let py = y + dy;

                    if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                        let pixel = buffer.get_pixel_mut(px as u32, py as u32);

                        // Distance-based alpha for soft edges
                        let dist = ((dx * dx + dy * dy) as f32).sqrt() / size as f32;
                        let edge_alpha = (1.0 - dist).max(0.0) * alpha;

                        // Additive blend
                        pixel[0] =
                            (pixel[0] as f32 + p.color.r() as f32 * edge_alpha).min(255.0) as u8;
                        pixel[1] =
                            (pixel[1] as f32 + p.color.g() as f32 * edge_alpha).min(255.0) as u8;
                        pixel[2] =
                            (pixel[2] as f32 + p.color.b() as f32 * edge_alpha).min(255.0) as u8;
                    }
                }
            }
        }
    }
}
