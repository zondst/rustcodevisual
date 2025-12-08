//! Offscreen Renderer for Video Export
//! High-quality software rendering to pixel buffer - completely independent of UI

use crate::audio::AudioState;
use crate::config::AppConfig;
use crate::config::BlendMode;
use crate::particles::ParticleEngine;
use egui::Color32;

/// Frame renderer for video export - high quality software rendering
pub struct FrameRenderer {
    width: u32,
    height: u32,
    buffer: Vec<u8>,
    // Temporary float buffer for HDR/additive blending
    hdr_buffer: Vec<f32>,
}

impl FrameRenderer {
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            width,
            height,
            buffer: vec![0u8; size * 4],
            hdr_buffer: vec![0.0f32; size * 3],
        }
    }

    #[allow(dead_code)]
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let size = (width * height) as usize;
        self.buffer = vec![0u8; size * 4];
        self.hdr_buffer = vec![0.0f32; size * 3];
    }

    /// Get buffer dimensions
    #[allow(dead_code)]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get raw RGBA buffer
    #[allow(dead_code)]
    pub fn get_buffer(&self) -> &[u8] {
        &self.buffer
    }

    /// Clear HDR buffer with background color
    fn clear_hdr(&mut self, bg_color: Color32) {
        let bg_r = bg_color.r() as f32;
        let bg_g = bg_color.g() as f32;
        let bg_b = bg_color.b() as f32;

        for i in 0..(self.width * self.height) as usize {
            self.hdr_buffer[i * 3] = bg_r;
            self.hdr_buffer[i * 3 + 1] = bg_g;
            self.hdr_buffer[i * 3 + 2] = bg_b;
        }
    }

    /// Convert HDR buffer to RGBA output with tone mapping
    fn tonemap_to_output(&mut self) {
        for i in 0..(self.width * self.height) as usize {
            // Simple reinhard-ish tone mapping for bloom
            let r = self.hdr_buffer[i * 3];
            let g = self.hdr_buffer[i * 3 + 1];
            let b = self.hdr_buffer[i * 3 + 2];

            // Soft clamp to prevent harsh clipping
            self.buffer[i * 4] = (r.min(255.0)) as u8;
            self.buffer[i * 4 + 1] = (g.min(255.0)) as u8;
            self.buffer[i * 4 + 2] = (b.min(255.0)) as u8;
            self.buffer[i * 4 + 3] = 255;
        }
    }

    /// Render a complete frame for export with HIGH QUALITY
    pub fn render_frame(
        &mut self,
        particles: &ParticleEngine,
        config: &AppConfig,
        _audio: &AudioState,
    ) -> &[u8] {
        // Get background color from config
        let colors = config.get_color_scheme();
        let bg_color = Color32::from_rgb(
            colors.background[0],
            colors.background[1],
            colors.background[2],
        );

        // Clear HDR buffer
        self.clear_hdr(bg_color);

        let glow_intensity = config.particles.glow_intensity;
        let use_volumetric = config.particles.volumetric_rendering;
        let blend_mode = config.particles.blend_mode.clone();

        // Render particles with GLOW effect
        for p in particles.get_particles().iter() {
            if p.audio_alpha < 0.01 || p.life <= 0.0 {
                continue;
            }

            let x = p.pos.x;
            let y = p.pos.y;

            // Use larger radius for glow effect (2-3x the particle size)
            let base_radius = p.size * p.audio_size;
            let glow_radius = base_radius * (1.5 + glow_intensity);
            let alpha = p.audio_alpha * p.brightness;

            if use_volumetric {
                // Multi-layer glow for smooth volumetric look
                self.draw_volumetric_glow(
                    x,
                    y,
                    glow_radius,
                    p.color,
                    alpha * glow_intensity,
                    &blend_mode,
                );
            }

            // Draw main particle
            self.draw_glow_particle(x, y, base_radius, p.color, alpha, &blend_mode);
        }

        // Convert to output
        self.tonemap_to_output();

        &self.buffer
    }

    /// Draw volumetric glow (outer layer)
    fn draw_volumetric_glow(
        &mut self,
        cx: f32,
        cy: f32,
        r: f32,
        color: Color32,
        alpha: f32,
        blend: &BlendMode,
    ) {
        if r <= 1.0 || alpha < 0.01 {
            return;
        }

        let w = self.width as i32;
        let h = self.height as i32;

        let _r_i = r as i32 + 2;
        let min_x = ((cx - r) as i32 - 2).max(0);
        let max_x = ((cx + r) as i32 + 2).min(w - 1);
        let min_y = ((cy - r) as i32 - 2).max(0);
        let max_y = ((cy + r) as i32 + 2).min(h - 1);

        let r_sq = r * r;

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let dx = px as f32 - cx;
                let dy = py as f32 - cy;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq <= r_sq {
                    let dist = dist_sq.sqrt();
                    // Smooth falloff with cubic easing
                    let t = 1.0 - (dist / r);
                    let falloff = t * t * (3.0 - 2.0 * t); // smoothstep
                    let a = alpha * falloff * 0.3; // Glow is more subtle

                    if a > 0.001 {
                        let idx = (py as usize * self.width as usize + px as usize) * 3;

                        match blend {
                            BlendMode::Add => {
                                self.hdr_buffer[idx] += color.r() as f32 * a;
                                self.hdr_buffer[idx + 1] += color.g() as f32 * a;
                                self.hdr_buffer[idx + 2] += color.b() as f32 * a;
                            }
                            _ => {
                                self.hdr_buffer[idx] =
                                    self.hdr_buffer[idx] * (1.0 - a) + color.r() as f32 * a;
                                self.hdr_buffer[idx + 1] =
                                    self.hdr_buffer[idx + 1] * (1.0 - a) + color.g() as f32 * a;
                                self.hdr_buffer[idx + 2] =
                                    self.hdr_buffer[idx + 2] * (1.0 - a) + color.b() as f32 * a;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Draw main particle with glow
    fn draw_glow_particle(
        &mut self,
        cx: f32,
        cy: f32,
        r: f32,
        color: Color32,
        alpha: f32,
        blend: &BlendMode,
    ) {
        if r <= 0.5 || alpha < 0.01 {
            return;
        }

        let w = self.width as i32;
        let h = self.height as i32;

        // Extend radius for smooth edge
        let extended_r = r * 1.5;
        let min_x = ((cx - extended_r) as i32).max(0);
        let max_x = ((cx + extended_r) as i32).min(w - 1);
        let min_y = ((cy - extended_r) as i32).max(0);
        let max_y = ((cy + extended_r) as i32).min(h - 1);

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let dx = px as f32 - cx;
                let dy = py as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist <= extended_r {
                    // Core is solid, edge fades smoothly
                    let core_factor = if dist < r * 0.5 {
                        1.0
                    } else if dist < r {
                        let t = (dist - r * 0.5) / (r * 0.5);
                        1.0 - t * t
                    } else {
                        let t = (dist - r) / (extended_r - r);
                        let fade = 1.0 - t;
                        fade * fade * 0.5
                    };

                    let a = alpha * core_factor;

                    if a > 0.001 {
                        let idx = (py as usize * self.width as usize + px as usize) * 3;

                        match blend {
                            BlendMode::Add => {
                                self.hdr_buffer[idx] += color.r() as f32 * a;
                                self.hdr_buffer[idx + 1] += color.g() as f32 * a;
                                self.hdr_buffer[idx + 2] += color.b() as f32 * a;
                            }
                            _ => {
                                self.hdr_buffer[idx] =
                                    self.hdr_buffer[idx] * (1.0 - a) + color.r() as f32 * a;
                                self.hdr_buffer[idx + 1] =
                                    self.hdr_buffer[idx + 1] * (1.0 - a) + color.g() as f32 * a;
                                self.hdr_buffer[idx + 2] =
                                    self.hdr_buffer[idx + 2] * (1.0 - a) + color.b() as f32 * a;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// HEADLESS EXPORT ENGINE - Runs completely separate from UI
// ============================================================================

use crate::audio::AdaptiveAudioNormalizer;
use crate::audio::AudioAnalysis;
use crate::export::VideoExporter;
use std::sync::mpsc::Sender;

/// Message to send export progress
pub enum ExportMessage {
    Progress(usize, usize), // current frame, total frames
    Completed,
    Error(String),
}

/// Run export in a completely separate context from UI
/// This function should be called from a separate thread
pub fn run_headless_export(
    config: AppConfig,
    audio_analysis: AudioAnalysis,
    output_path: std::path::PathBuf,
    duration_secs: f32,
    progress_tx: Sender<ExportMessage>,
) {
    // Create our own instances - completely independent from UI
    let width = config.export.width;
    let height = config.export.height;
    let fps = config.export.fps;

    let total_frames = (duration_secs * fps as f32) as usize;
    let dt = 1.0 / fps as f32;

    // Our own particle engine
    let mut particles = ParticleEngine::new(width as f32, height as f32);
    particles.update_palette(&config.get_color_scheme());

    // Our own audio state
    let mut audio_state = crate::audio::AudioState::new();

    // Our own normalizer
    let mut audio_normalizer =
        AdaptiveAudioNormalizer::new(config.particles.adaptive_window_secs, fps);

    // Our own frame renderer
    let mut frame_renderer = FrameRenderer::new(width, height);

    // Our own video exporter
    let mut video_exporter = VideoExporter::new();
    video_exporter.format = match config.export.codec.as_str() {
        "libx264" => crate::export::ExportFormat::MP4,
        _ => crate::export::ExportFormat::MP4,
    };

    // Start exporter
    if let Err(e) = video_exporter.start(output_path, width, height, fps, total_frames) {
        let _ = progress_tx.send(ExportMessage::Error(e));
        return;
    }

    // Main export loop - as fast as possible!
    for frame_idx in 0..total_frames {
        // 1. Get audio frame
        let audio_frame_idx = (frame_idx as f32 * audio_analysis.fps as f32 / fps as f32) as usize;
        if audio_frame_idx < audio_analysis.total_frames {
            let frame = audio_analysis.get_frame(audio_frame_idx);
            audio_state.update_from_frame(&frame, config.audio.smoothing);
        }

        // 2. Update smoothing
        audio_state.update_smoothing(
            dt,
            config.audio.smoothing,
            config.audio.beat_attack,
            config.audio.beat_decay,
        );

        // 3. Update adaptive normalization
        let normalized = if config.particles.adaptive_audio_enabled {
            Some(audio_normalizer.normalize(
                audio_state.smooth_bass,
                audio_state.smooth_mid,
                audio_state.smooth_high,
                config.particles.bass_sensitivity,
                config.particles.mid_sensitivity,
                config.particles.high_sensitivity,
                config.particles.adaptive_strength,
            ))
        } else {
            None
        };

        // 4. Update particles
        particles.update(&config.particles, &audio_state, dt, normalized.as_ref());

        // 5. Render frame
        let frame_data = frame_renderer.render_frame(&particles, &config, &audio_state);

        // 6. Write to video
        if let Err(e) = video_exporter.write_frame(frame_data) {
            let _ = progress_tx.send(ExportMessage::Error(format!("Write error: {}", e)));
            return;
        }

        // 7. Send progress every 10 frames (reduce overhead)
        if frame_idx % 10 == 0 {
            let _ = progress_tx.send(ExportMessage::Progress(frame_idx, total_frames));
        }
    }

    // Finish
    if let Err(e) = video_exporter.finish() {
        let _ = progress_tx.send(ExportMessage::Error(e));
        return;
    }

    let _ = progress_tx.send(ExportMessage::Completed);
}
