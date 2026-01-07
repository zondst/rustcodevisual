//! CPU (headless) offscreen renderer + exporter.
//!
//! This module is used for the **CPU export** path (headless rendering).
//! The goal is to be deterministic and to stay as close as practical to the preview logic.
//!
//! Key features:
//! - Export format is explicit (MP4 / MOV Alpha / WebM Alpha / PNG sequence)
//! - Optional audio muxing (so the render is "под музыку")
//! - Overlay (waveform + spectrum) rendered consistently (same CPU overlay used by GPU export)
//! - Meaningful alpha channel for alpha-capable formats (transparent background)
//!
//! Notes about visual parity:
//! - The preview is rendered via egui (vector-ish) while headless export is CPU rasterized.
//!   Expect small differences, but the color pipeline + alpha + overlays should be consistent.

use crate::audio::{AdaptiveAudioNormalizer, AudioAnalysis, AudioState, NormalizedAudio};
use crate::config::{AppConfig, BlendMode, ParticleMode};
use crate::export::{ExportFormat, VideoExporter};
use crate::gpu_export; // CPU overlay renderer (waveform + spectrum)
use crate::particles::ParticleEngine;

use egui::Color32;

use std::path::PathBuf;
use std::sync::mpsc::Sender;

/// Messages from the export thread to the UI.
///
/// NOTE: This must match the pattern matching in `main.rs`.
#[derive(Debug, Clone)]
pub enum ExportMessage {
    /// (current_frame, total_frames)
    Progress(usize, usize),
    Completed,
    Error(String),
}

/// Simple CPU renderer.
///
/// It renders into float RGB + float alpha buffers, then quantizes to RGBA8 for encoding.
pub struct FrameRenderer {
    width: u32,
    height: u32,
    /// RGB buffer in 0..255-ish float space (can exceed 255 for additive glow).
    rgb: Vec<f32>,
    /// Alpha buffer in 0..1.
    alpha: Vec<f32>,
    /// Output RGBA8.
    out_rgba: Vec<u8>,
}

impl FrameRenderer {
    pub fn new(width: u32, height: u32) -> Self {
        let px = (width * height) as usize;
        Self {
            width,
            height,
            rgb: vec![0.0; px * 3],
            alpha: vec![0.0; px],
            out_rgba: vec![0; px * 4],
        }
    }

    fn clear(&mut self, bg: Color32, bg_alpha: f32) {
        let r = bg.r() as f32;
        let g = bg.g() as f32;
        let b = bg.b() as f32;

        let a = bg_alpha.clamp(0.0, 1.0);
        let px = (self.width * self.height) as usize;
        for i in 0..px {
            let base = i * 3;
            self.rgb[base] = r;
            self.rgb[base + 1] = g;
            self.rgb[base + 2] = b;
            self.alpha[i] = a;
        }
    }

    /// Blend a full-frame RGBA8 overlay over the current buffer.
    ///
    /// Overlay is expected to be **straight alpha** RGBA8 (same as produced by
    /// `gpu_export::render_overlay_cpu`).
    fn blend_overlay(&mut self, overlay_rgba: &[u8]) {
        debug_assert_eq!(overlay_rgba.len(), self.out_rgba.len());
        let px = (self.width * self.height) as usize;
        for i in 0..px {
            let o = i * 4;
            let oa = overlay_rgba[o + 3] as f32 / 255.0;
            if oa <= 0.0 {
                continue;
            }
            let or = overlay_rgba[o] as f32;
            let og = overlay_rgba[o + 1] as f32;
            let ob = overlay_rgba[o + 2] as f32;

            // Straight alpha OVER in RGB space.
            let base = i * 3;
            self.rgb[base] = or * oa + self.rgb[base] * (1.0 - oa);
            self.rgb[base + 1] = og * oa + self.rgb[base + 1] * (1.0 - oa);
            self.rgb[base + 2] = ob * oa + self.rgb[base + 2] * (1.0 - oa);

            // Alpha OVER
            let da = self.alpha[i];
            self.alpha[i] = oa + da * (1.0 - oa);
        }
    }

    fn draw_circle_soft(
        &mut self,
        cx: f32,
        cy: f32,
        radius: f32,
        color: Color32,
        alpha: f32,
        blend_mode: BlendMode,
    ) {
        if radius <= 0.1 || alpha <= 0.0 {
            return;
        }

        let min_x = (cx - radius).floor().max(0.0) as i32;
        let max_x = (cx + radius).ceil().min(self.width as f32 - 1.0) as i32;
        let min_y = (cy - radius).floor().max(0.0) as i32;
        let max_y = (cy + radius).ceil().min(self.height as f32 - 1.0) as i32;

        let r = color.r() as f32;
        let g = color.g() as f32;
        let b = color.b() as f32;

        let radius_sq = radius * radius;

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let dx = px as f32 - cx;
                let dy = py as f32 - cy;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq > radius_sq {
                    continue;
                }

                // Smooth falloff (a bit nicer than linear)
                let dist = dist_sq.sqrt();
                let t = (dist / radius).clamp(0.0, 1.0);
                // smoothstep-like curve
                let falloff = (1.0 - t).powf(1.8);
                let a = (alpha * falloff).clamp(0.0, 1.0);

                let idx = (py as u32 * self.width + px as u32) as usize;
                let base = idx * 3;

                match blend_mode {
                    BlendMode::Add => {
                        self.rgb[base] += r * a;
                        self.rgb[base + 1] += g * a;
                        self.rgb[base + 2] += b * a;
                    }
                    // Approximate other modes as normal OVER
                    _ => {
                        self.rgb[base] = self.rgb[base] * (1.0 - a) + r * a;
                        self.rgb[base + 1] = self.rgb[base + 1] * (1.0 - a) + g * a;
                        self.rgb[base + 2] = self.rgb[base + 2] * (1.0 - a) + b * a;
                    }
                }

                // Alpha coverage (OVER)
                let da = self.alpha[idx];
                self.alpha[idx] = a + da * (1.0 - a);
            }
        }
    }

    /// Volumetric particle rendering inspired by `ParticleEngine::draw_volumetric_particle`.
    fn draw_volumetric_particle(
        &mut self,
        x: f32,
        y: f32,
        size: f32,
        base_color: Color32,
        base_alpha_u8: u8,
        config: &crate::config::ParticleConfig,
        blend_mode: BlendMode,
    ) {
        let base_radius = size.max(0.1);
        let glow_radius = base_radius * (1.0 + config.glow_intensity * 2.0);
        let steps = config.volumetric_steps.clamp(8, 24) as usize;

        let base_alpha = base_alpha_u8 as f32 / 255.0;

        // Outer -> inner (same direction as preview)
        for i in (0..steps).rev() {
            let t = i as f32 / steps as f32; // 0..1
            let radius = base_radius + (glow_radius - base_radius) * t;

            // Gaussian falloff
            let dist_norm = t;
            let falloff = (-2.5 * dist_norm * dist_norm).exp();

            // Core boost
            let intensity_boost = if t < 0.2 { 1.5 } else { 1.0 };

            // Distribute energy across steps
            let layer_alpha = (base_alpha * falloff * intensity_boost * (1.0 / steps as f32) * 2.0)
                .clamp(0.0, 1.0);

            if layer_alpha <= 0.002 {
                continue;
            }

            self.draw_circle_soft(x, y, radius, base_color, layer_alpha, blend_mode);
        }

        // Hot core (white-ish)
        let core_color = Color32::from_rgba_unmultiplied(255, 255, 255, 200);
        self.draw_circle_soft(x, y, size * 0.2, core_color, (200.0 / 255.0) * base_alpha, blend_mode);
    }

    fn render_frame(
        &mut self,
        particles: &ParticleEngine,
        config: &AppConfig,
        audio_state: &AudioState,
        normalized: Option<&NormalizedAudio>,
        transparent_bg: bool,
    ) -> &[u8] {
        let colors = config.get_color_scheme();

        let bg_color = if transparent_bg {
            Color32::from_rgba_unmultiplied(0, 0, 0, 0)
        } else {
            Color32::from_rgb(colors.background[0], colors.background[1], colors.background[2])
        };
        let bg_alpha = if transparent_bg { 0.0 } else { 1.0 };

        self.clear(bg_color, bg_alpha);

        // Overlay (spectrum + waveform) behind particles, like in preview.
        let overlay = gpu_export::render_overlay_cpu(self.width, self.height, audio_state, config);
        self.blend_overlay(&overlay);

        // Effective amplitude for subtle size breathing (matching preview)
        let eff_amp = if config.particles.adaptive_audio_enabled {
            normalized.map(|n| n.intensity).unwrap_or(audio_state.amplitude)
        } else {
            audio_state.amplitude
        };

        // Particles
        let blend_mode = config.particles.blend_mode;
        for p in particles.get_particles_iter() {
            let life_alpha = (p.life / p.max_life).clamp(0.0, 1.0);
            let audio_factor = if config.particles.audio_reactive_spawn {
                p.audio_alpha
            } else {
                1.0
            };

            let alpha_f = (life_alpha * p.brightness * audio_factor).clamp(0.0, 1.0);
            let alpha_u8 = (alpha_f * 255.0).min(255.0) as u8;
            if alpha_u8 < 3 {
                continue;
            }

            let mut size = p.size * (1.0 + eff_amp * 0.2);
            // Glow particles are a bit larger in preview
            if config.particles.shape == crate::config::ParticleShape::Glow {
                size *= 1.5;
            }

            // Use the same volumetric approach as preview when enabled; otherwise fallback to a soft circle.
            if config.particles.volumetric_rendering {
                self.draw_volumetric_particle(p.pos.x, p.pos.y, size, p.color, alpha_u8, &config.particles, blend_mode);
            } else {
                self.draw_circle_soft(p.pos.x, p.pos.y, size, p.color, alpha_f, blend_mode);
            }
        }

        // Quantize (simple tonemap + exposure)
        let px = (self.width * self.height) as usize;

        // Use the same exposure convention as GPU tonemap (stops): multiply by 2^exposure.
        // If you prefer linear exposure, replace with: `let exposure_mult = config.visual.exposure;`
        let exposure_mult = 2.0_f32.powf(config.visual.exposure);

        for i in 0..px {
            let base = i * 3;
            let mut r = (self.rgb[base] / 255.0) * exposure_mult;
            let mut g = (self.rgb[base + 1] / 255.0) * exposure_mult;
            let mut b = (self.rgb[base + 2] / 255.0) * exposure_mult;

            // Gentle tonemap to avoid harsh clipping
            r = r / (1.0 + r);
            g = g / (1.0 + g);
            b = b / (1.0 + b);

            let a = if transparent_bg {
                self.alpha[i].clamp(0.0, 1.0)
            } else {
                1.0
            };

            let o = i * 4;
            self.out_rgba[o] = (r.clamp(0.0, 1.0) * 255.0) as u8;
            self.out_rgba[o + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
            self.out_rgba[o + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
            self.out_rgba[o + 3] = (a * 255.0) as u8;
        }

        &self.out_rgba
    }
}

/// Run headless CPU export in the current thread.
///
/// `main.rs` typically spawns this in a background thread.
pub fn run_headless_export(
    config: AppConfig,
    audio_analysis: AudioAnalysis,
    output_path: PathBuf,
    audio_path: Option<PathBuf>,
    export_format: ExportFormat,
    start_time_secs: f32,
    duration: f32,
    progress_tx: Sender<ExportMessage>,
) {
    let width = config.export.width;
    let height = config.export.height;
    let fps = config.export.fps;

    let total_frames = (duration * fps as f32).ceil() as usize;
    let start_frame = (start_time_secs.max(0.0) * fps as f32).round() as usize;
    let total_sim_frames = start_frame + total_frames;
    if total_frames == 0 {
        let _ = progress_tx.send(ExportMessage::Error("Export duration is 0".to_string()));
        return;
    }

    // Start exporter
    let mut exporter = VideoExporter::new();
    exporter.format = export_format;
    exporter.mp4_crf = config.export.crf;
    exporter.mp4_preset = config.export.preset.clone();

    if let Err(e) = exporter.start(
        output_path.clone(),
        width,
        height,
        fps,
        total_frames,
        audio_path.clone(),
    ) {
        let _ = progress_tx.send(ExportMessage::Error(e));
        return;
    }

    // Engine state
    let mut particles = ParticleEngine::new(width as f32, height as f32);
    let colors = config.get_color_scheme();
    particles.update_palette(&colors);

    let mut audio_state = AudioState::new();
    let dt = 1.0 / fps as f32;

    // Adaptive audio normalization (matches preview behaviour)
    let mut audio_normalizer = AdaptiveAudioNormalizer::new(config.particles.adaptive_window_secs, fps);
    let mut normalized_audio = NormalizedAudio::default();

    // Renderer
    let mut renderer = FrameRenderer::new(width, height);
    let transparent_bg = export_format.supports_alpha();

    for sim_frame_idx in 0..total_sim_frames {
        // Map export frame -> audio analysis frame (supports export fps != analysis fps)
        let audio_frame_idx =
            ((sim_frame_idx as f32) * (audio_analysis.fps as f32 / fps as f32)).floor() as usize;

        let frame = audio_analysis.get_frame(audio_frame_idx);
        audio_state.update_from_frame(&frame, config.audio.smoothing);
        audio_state.update_smoothing(
            dt,
            config.audio.smoothing,
            config.audio.beat_attack,
            config.audio.beat_decay,
        );

        if config.particles.adaptive_audio_enabled {
            normalized_audio = audio_normalizer.normalize(
                audio_state.smooth_bass,
                audio_state.smooth_mid,
                audio_state.smooth_high,
                config.particles.bass_sensitivity,
                config.particles.mid_sensitivity,
                config.particles.high_sensitivity,
                config.particles.adaptive_strength,
            );
        }

        let normalized_ref = if config.particles.adaptive_audio_enabled {
            Some(&normalized_audio)
        } else {
            None
        };

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

        if sim_frame_idx < start_frame {
            continue;
        }
        let out_frame_idx = sim_frame_idx - start_frame;

        let rgba = renderer.render_frame(&particles, &config, &audio_state, normalized_ref, transparent_bg);

        if let Err(e) = exporter.write_frame(rgba) {
            let _ = progress_tx.send(ExportMessage::Error(e));
            let _ = exporter.finish();
            return;
        }

        let _ = progress_tx.send(ExportMessage::Progress(out_frame_idx + 1, total_frames));
    }

    if let Err(e) = exporter.finish() {
        let _ = progress_tx.send(ExportMessage::Error(e));
        return;
    }

    let _ = progress_tx.send(ExportMessage::Completed);
}
