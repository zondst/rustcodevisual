//! Particle Engine for Particle Studio RS
//! Advanced particle system with physics, audio reactivity, and multiple modes

use crate::config::{ParticleConfig, ParticleMode, ParticleShape, ColorScheme};
use crate::audio::{AudioState, NormalizedAudio};
use egui::{Color32, Painter, Pos2, Vec2, Rect};
use rand::Rng;
use std::f32::consts::PI;

/// Individual particle data
#[derive(Clone)]
pub struct Particle {
    pub pos: Vec2,
    pub vel: Vec2,
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
    pub base_size: f32,
    pub color: Color32,
    pub color_index: usize,
    pub angle: f32,
    pub angular_vel: f32,
    pub brightness: f32,
    // Audio-driven state
    pub audio_alpha: f32,     // Opacity driven by audio (0.0-1.0)
    pub audio_size: f32,      // Size multiplier from audio
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            pos: Vec2::ZERO,
            vel: Vec2::ZERO,
            life: 1.0,
            max_life: 1.0,
            size: 5.0,
            base_size: 5.0,
            color: Color32::WHITE,
            color_index: 0,
            angle: 0.0,
            angular_vel: 0.0,
            brightness: 1.0,
            audio_alpha: 0.0,
            audio_size: 1.0,
        }
    }
}

/// Particle Engine managing all particles
pub struct ParticleEngine {
    pub particles: Vec<Particle>,
    pub width: f32,
    pub height: f32,
    
    // Physics state
    time: f32,
    flow_field_time: f32,
    
    // Palette cache
    palette: Vec<Color32>,
}

impl ParticleEngine {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            particles: Vec::with_capacity(10000),
            width,
            height,
            time: 0.0,
            flow_field_time: 0.0,
            palette: vec![Color32::WHITE],
        }
    }
    
    pub fn update_palette(&mut self, colors: &ColorScheme) {
        self.palette = colors.particles
            .iter()
            .map(|c| Color32::from_rgb(c[0], c[1], c[2]))
            .collect();
        
        if self.palette.is_empty() {
            self.palette.push(Color32::WHITE);
        }
    }
    
    /// Get read-only access to particles for rendering
    pub fn get_particles(&self) -> &Vec<Particle> {
        &self.particles
    }
    
    pub fn update(&mut self, config: &ParticleConfig, audio: &AudioState, dt: f32, normalized: Option<&NormalizedAudio>) {
        if !config.enabled {
            self.particles.clear();
            return;
        }
        
        self.time += dt;
        self.flow_field_time += dt * 0.5;
        
        let mut rng = rand::thread_rng();
        
        // ========================================================
        // RULE 1: Detect audio state with adaptive normalization
        // ========================================================
        let (audio_level, has_audio, is_beat, is_strong_beat) = if config.adaptive_audio_enabled {
            if let Some(norm) = normalized {
                // Adaptive mode: use normalized values relative to track dynamics
                let level = norm.intensity;
                let has = norm.has_significant_audio;
                let beat = norm.has_bass_hit;
                let strong = norm.has_bass_hit && norm.bass > 0.6;
                (level, has, beat, strong)
            } else {
                // Fallback if no normalized data
                let level = audio.smooth_amplitude;
                let has = level > config.audio_spawn_threshold;
                (level, has, audio.smooth_beat > 0.5, audio.smooth_beat > 0.7)
            }
        } else {
            // Classic mode: fixed threshold
            let level = audio.smooth_amplitude;
            let has = level > config.audio_spawn_threshold;
            (level, has, audio.smooth_beat > 0.5, audio.smooth_beat > 0.7)
        };
        
        let center = Vec2::new(self.width / 2.0, self.height / 2.0);
        
        // ========================================================
        // RULE 2: Spawn particles ONLY when there's audio
        // ========================================================
        if config.audio_reactive_spawn {
            if has_audio {
                // Spawn rate proportional to audio level
                let spawn_rate = (audio_level * config.count as f32 * 0.08 * dt * 60.0) as usize;
                
                // Extra burst on beats
                let beat_spawn = if is_beat { 
                    (config.count as f32 * 0.03) as usize 
                } else { 
                    0 
                };
                
                let to_spawn = (spawn_rate + beat_spawn).min(15);
                for _ in 0..to_spawn {
                    if self.particles.len() < config.count {
                        self.spawn_audio_particle(config, audio, &mut rng, center);
                    }
                }
            }
            // When no audio: don't spawn, let existing particles fade
        } else {
            // Non-audio-reactive mode: standard spawning
            while self.particles.len() < config.count {
                self.spawn_particle(config, &mut rng);
            }
            if self.particles.len() > config.count {
                self.particles.truncate(config.count);
            }
        }
        
        let speed_factor = config.speed * 60.0 * dt;
        
        // ========================================================
        // RULE 3: Update all particles with audio-physics
        // ========================================================
        for p in &mut self.particles {
            // -- BEAT PHYSICS: Velocity burst away from center --
            if is_strong_beat && config.beat_burst_strength > 0.0 {
                let dir = Vec2::new(p.pos.x - center.x, p.pos.y - center.y);
                let dist = (dir.x * dir.x + dir.y * dir.y).sqrt().max(1.0);
                let normalized = Vec2::new(dir.x / dist, dir.y / dist);
                
                // Burst velocity proportional to beat strength
                let burst = audio.smooth_beat * config.beat_burst_strength * 0.3;
                p.vel.x += normalized.x * burst;
                p.vel.y += normalized.y * burst;
            }
            
            // -- SIZE: Driven by audio amplitude --
            if config.audio_reactive_spawn {
                p.audio_size = audio_level.clamp(0.3, 1.5);
                let beat_pulse = 1.0 + audio.smooth_beat * config.beat_size_pulse * 0.2;
                p.size = p.base_size * p.audio_size * beat_pulse;
            } else {
                let size_pulse = 1.0 + audio.smooth_beat * config.beat_size_pulse * 0.3;
                p.size = p.base_size * size_pulse;
            }
            
            // -- OPACITY: Fade with audio (asymmetric attack/release) --
            if config.audio_reactive_spawn {
                let target_alpha = if has_audio {
                    audio_level.clamp(0.4, 1.0)
                } else {
                    0.0  // Fade to invisible when no audio
                };
                
                // Asymmetric fade: fast appear, slow fade
                let fade_speed = if target_alpha > p.audio_alpha {
                    config.fade_attack_speed
                } else {
                    config.fade_release_speed
                };
                
                p.audio_alpha += (target_alpha - p.audio_alpha) * fade_speed * dt;
                p.audio_alpha = p.audio_alpha.clamp(0.0, 1.0);
            } else {
                p.audio_alpha = 1.0;  // Always visible in non-reactive mode
            }
            
            // -- MOVEMENT --
            p.pos += p.vel * speed_factor;
            
            // Damping (exponential)
            let damping = (-config.damping * dt).exp();
            p.vel *= damping;
            
            // Clamp max velocity
            let vel_mag = (p.vel.x * p.vel.x + p.vel.y * p.vel.y).sqrt();
            if vel_mag > 5.0 {
                let scale = 5.0 / vel_mag;
                p.vel.x *= scale;
                p.vel.y *= scale;
            }
            
            // Update angle
            p.angle += p.angular_vel * dt;
            
            // Decrease life
            p.life -= dt;
            
            // Brightness from audio
            p.brightness = if has_audio {
                0.7 + audio_level * 0.2 + audio.smooth_beat * 0.1
            } else {
                0.5
            };
            
            // Wrap around screen
            if p.pos.x < -50.0 { p.pos.x = self.width + 50.0; }
            if p.pos.x > self.width + 50.0 { p.pos.x = -50.0; }
            if p.pos.y < -50.0 { p.pos.y = self.height + 50.0; }
            if p.pos.y > self.height + 50.0 { p.pos.y = -50.0; }
        }
        
        // ========================================================
        // RULE 4: Remove particles when fully faded or dead
        // ========================================================
        if config.audio_reactive_spawn {
            self.particles.retain(|p| p.audio_alpha > 0.01 && p.life > 0.0);
        } else {
            // Non-reactive mode: reset dead particles
            let mut dead_indices: Vec<usize> = self.particles.iter()
                .enumerate()
                .filter(|(_, p)| p.life <= 0.0)
                .map(|(i, _)| i)
                .collect();
            
            for idx in dead_indices {
                if idx < self.particles.len() {
                    self.reset_particle_at(idx, config, &mut rng);
                }
            }
        }
    }
    
    /// Spawn a particle driven by audio (from center, bursting outward)
    fn spawn_audio_particle(&mut self, config: &ParticleConfig, audio: &AudioState, rng: &mut impl Rng, center: Vec2) {
        let angle = rng.gen::<f32>() * std::f32::consts::TAU;
        
        // Spawn position: center or random based on config
        let pos = if config.spawn_from_center {
            let radius = rng.gen::<f32>() * config.spawn_radius + 10.0;
            Vec2::new(
                center.x + angle.cos() * radius,
                center.y + angle.sin() * radius,
            )
        } else {
            Vec2::new(
                rng.gen::<f32>() * self.width,
                rng.gen::<f32>() * self.height,
            )
        };
        
        // Initial velocity: outward from center, scaled by audio
        let speed = audio.smooth_amplitude * 0.5 + 0.1;
        let vel = Vec2::new(angle.cos() * speed, angle.sin() * speed);
        
        let color_idx = rng.gen_range(0..self.palette.len());
        let base_size = config.min_size + rng.gen::<f32>() * (config.max_size - config.min_size);
        
        let particle = Particle {
            pos,
            vel,
            life: 2.0 + rng.gen::<f32>() * 3.0,
            max_life: 5.0,
            size: base_size * audio.smooth_amplitude.clamp(0.3, 1.0),
            base_size,
            color: self.palette[color_idx],
            color_index: color_idx,
            angle: rng.gen::<f32>() * std::f32::consts::TAU,
            angular_vel: (rng.gen::<f32>() - 0.5) * 2.0,
            brightness: 1.0,
            audio_alpha: 0.1,  // Start low, will ramp up
            audio_size: audio.smooth_amplitude.clamp(0.3, 1.0),
        };
        
        self.particles.push(particle);
    }
    
    fn update_chaos(&mut self, config: &ParticleConfig, audio: &AudioState, dt: f32, speed_factor: f32) {
        let center = Vec2::new(self.width / 2.0, self.height / 2.0);
        
        for p in &mut self.particles {
            // Curl noise flow field
            let noise_scale = 0.003;
            let nx = p.pos.x * noise_scale + self.flow_field_time;
            let ny = p.pos.y * noise_scale + self.flow_field_time * 0.7;
            
            let flow_x = (ny * 6.0).sin() + (nx * 3.0).cos() * 0.5;
            let flow_y = (nx * 6.0).cos() + (ny * 3.0).sin() * 0.5;
            
            p.vel.x += flow_x * 0.1 * speed_factor * (1.0 + audio.mid);
            p.vel.y += flow_y * 0.1 * speed_factor * (1.0 + audio.mid);
            
            // Beat explosion
            if audio.beat > 0.5 {
                let dx = p.pos.x - center.x;
                let dy = p.pos.y - center.y;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);
                
                let force = audio.beat * config.speed * 2.0;
                p.vel.x += (dx / dist) * force;
                p.vel.y += (dy / dist) * force;
            }
            
            // Damping
            p.vel *= 0.98;
            
            // Gravity
            p.vel.y += config.gravity * speed_factor * 0.5;
        }
    }
    
    fn update_calm(&mut self, config: &ParticleConfig, audio: &AudioState, dt: f32, speed_factor: f32) {
        for p in &mut self.particles {
            // Gentle floating motion
            let float_x = (self.time * 0.5 + p.pos.y * 0.01).sin() * 0.02;
            let float_y = -0.02 - audio.amplitude * 0.03;
            
            p.vel.x += float_x * speed_factor;
            p.vel.y += float_y * speed_factor;
            
            // Strong damping for calm effect
            p.vel *= 0.96;
            
            // Gravity (upward drift in calm mode)
            p.vel.y -= 0.01 * speed_factor;
        }
    }
    
    fn update_cinematic(&mut self, config: &ParticleConfig, audio: &AudioState, dt: f32, speed_factor: f32) {
        for p in &mut self.particles {
            // Very slow, smooth motion
            let breathing = (self.time * 0.3 + p.pos.x * 0.001).sin();
            
            p.vel.x += breathing * 0.005 * speed_factor;
            p.vel.y += (self.time * 0.2).cos() * 0.003 * speed_factor;
            
            // High damping
            p.vel *= 0.95;
            
            // Size breathing effect is more pronounced
            p.size = p.base_size * (1.0 + breathing * 0.3 + audio.amplitude * 0.4);
        }
    }
    
    fn update_orbit(&mut self, config: &ParticleConfig, audio: &AudioState, dt: f32, speed_factor: f32) {
        let center = Vec2::new(self.width / 2.0, self.height / 2.0);
        
        for p in &mut self.particles {
            let dx = p.pos.x - center.x;
            let dy = p.pos.y - center.y;
            let dist = (dx * dx + dy * dy).sqrt().max(1.0);
            
            // Perpendicular velocity (orbit)
            let perp_x = -dy / dist;
            let perp_y = dx / dist;
            
            let orbit_speed = config.orbit_speed * (1.0 + audio.beat * 0.5);
            p.vel.x += perp_x * orbit_speed * speed_factor;
            p.vel.y += perp_y * orbit_speed * speed_factor;
            
            // Pull toward center
            let pull_strength = 0.02;
            p.vel.x -= (dx / dist) * pull_strength * speed_factor;
            p.vel.y -= (dy / dist) * pull_strength * speed_factor;
            
            // Moderate damping
            p.vel *= 0.97;
        }
    }
    
    fn spawn_particle(&mut self, config: &ParticleConfig, rng: &mut rand::rngs::ThreadRng) {
        let mut p = Particle::default();
        self.reset_particle(&mut p, config, rng);
        self.particles.push(p);
    }
    
    fn reset_particle(&self, p: &mut Particle, config: &ParticleConfig, rng: &mut rand::rngs::ThreadRng) {
        // Random position
        p.pos = Vec2::new(
            rng.gen_range(0.0..self.width),
            rng.gen_range(0.0..self.height),
        );
        
        // Random velocity based on spread
        let angle = rng.gen_range(0.0..config.spread.to_radians());
        let speed = rng.gen_range(0.5..2.0);
        p.vel = Vec2::new(angle.cos() * speed, angle.sin() * speed);
        
        // Life
        p.life = rng.gen_range(0.5..2.0);
        p.max_life = p.life;
        
        // Size with variation (safe range handling)
        let size_range = (config.max_size - config.min_size).max(0.01);
        let base = config.min_size + rng.gen_range(0.0..size_range);
        let var = config.size_variation.max(0.01);
        let variation = 1.0 + rng.gen_range(-var..var);
        p.base_size = base * variation;
        p.size = p.base_size;
        
        // Color from palette
        p.color_index = rng.gen_range(0..self.palette.len());
        p.color = self.palette[p.color_index];
        
        // Rotation
        p.angle = rng.gen_range(0.0..PI * 2.0);
        p.angular_vel = rng.gen_range(-2.0..2.0);
        
        p.brightness = 1.0;
    }
    
    fn reset_particle_at(&mut self, idx: usize, config: &ParticleConfig, rng: &mut rand::rngs::ThreadRng) {
        if idx >= self.particles.len() {
            return;
        }
        
        let width = self.width;
        let height = self.height;
        let palette_len = self.palette.len();
        let palette = self.palette.clone();
        
        let p = &mut self.particles[idx];
        
        // Random position
        p.pos = Vec2::new(
            rng.gen_range(0.0..width),
            rng.gen_range(0.0..height),
        );
        
        // Random velocity based on spread
        let angle = rng.gen_range(0.0..config.spread.to_radians());
        let speed = rng.gen_range(0.5..2.0);
        p.vel = Vec2::new(angle.cos() * speed, angle.sin() * speed);
        
        // Life
        p.life = rng.gen_range(0.5..2.0);
        p.max_life = p.life;
        
        // Size with variation (safe range handling)
        let size_range = (config.max_size - config.min_size).max(0.01);
        let base = config.min_size + rng.gen_range(0.0..size_range);
        let var = config.size_variation.max(0.01);
        let variation = 1.0 + rng.gen_range(-var..var);
        p.base_size = base * variation;
        p.size = p.base_size;
        
        // Color from palette
        p.color_index = rng.gen_range(0..palette_len);
        p.color = palette[p.color_index];
        
        // Rotation
        p.angle = rng.gen_range(0.0..PI * 2.0);
        p.angular_vel = rng.gen_range(-2.0..2.0);
        
        p.brightness = 1.0;
    }
    
    /// Render particles to egui painter
    pub fn render(&self, painter: &Painter, rect: Rect, config: &ParticleConfig, audio: &AudioState) {
        if !config.enabled {
            return;
        }
        
        for p in &self.particles {
            // Use audio_alpha for audio-reactive visibility
            let life_alpha = (p.life / p.max_life).clamp(0.0, 1.0);
            let audio_factor = if config.audio_reactive_spawn { 
                p.audio_alpha 
            } else { 
                1.0 
            };
            let alpha = (life_alpha * p.brightness * audio_factor * 255.0).min(255.0) as u8;
            
            // Skip nearly invisible particles
            if alpha < 3 {
                continue;
            }
            
            let pos = rect.min + Vec2::new(p.pos.x, p.pos.y);
            
            // Skip if outside visible area (with margin for glow)
            let margin = p.size * 2.0;
            if pos.x < rect.left() - margin || pos.x > rect.right() + margin ||
               pos.y < rect.top() - margin || pos.y > rect.bottom() + margin {
                continue;
            }
            
            let color = Color32::from_rgba_premultiplied(
                p.color.r(),
                p.color.g(),
                p.color.b(),
                alpha,
            );
            
            let size = p.size * (1.0 + audio.amplitude * 0.2);
            
            // Render based on shape
            match config.shape {
                ParticleShape::Circle | ParticleShape::Point => {
                    if config.volumetric_rendering {
                        // Volumetric rendering with smooth radial gradient
                        self.draw_volumetric_particle(painter, pos, size, p.color, alpha, config);
                    } else {
                        // Basic circle with subtle gradient (legacy)
                        let glow_alpha = (alpha / 3).max(5);
                        let glow_color = Color32::from_rgba_premultiplied(
                            p.color.r(), p.color.g(), p.color.b(), glow_alpha
                        );
                        painter.circle_filled(pos, size * 1.2, glow_color);
                        painter.circle_filled(pos, size, color);
                    }
                }
                ParticleShape::Ring => {
                    painter.circle_stroke(pos, size, egui::Stroke::new(2.0, color));
                }
                ParticleShape::Glow => {
                    if config.volumetric_rendering {
                        // Volumetric rendering with more steps and larger radius
                        self.draw_volumetric_particle(painter, pos, size * 1.5, p.color, alpha, config);
                    } else {
                        // Old layer-based glow (legacy)
                        let intensity = config.glow_intensity.clamp(0.1, 1.0);
                        let base_alpha = (alpha as f32 * intensity) as u8;
                        
                        // Layer 1: Outer soft glow
                        let outer_alpha = (base_alpha / 10).max(2);
                        let outer_glow = Color32::from_rgba_premultiplied(
                            p.color.r(), p.color.g(), p.color.b(), outer_alpha
                        );
                        painter.circle_filled(pos, size * 1.6, outer_glow);
                        
                        // Layer 2: Mid glow
                        let mid_alpha = (base_alpha / 6).max(3);
                        let mid_glow = Color32::from_rgba_premultiplied(
                            p.color.r(), p.color.g(), p.color.b(), mid_alpha
                        );
                        painter.circle_filled(pos, size * 1.2, mid_glow);
                        
                        // Layer 3: Core
                        let core_alpha = (base_alpha / 2).max(10);
                        let core = Color32::from_rgba_premultiplied(
                            p.color.r(), p.color.g(), p.color.b(), core_alpha
                        );
                        painter.circle_filled(pos, size * 0.7, core);
                        
                        // Layer 4: Bright center
                        let center_alpha = base_alpha.min(150);
                        let center = Color32::from_rgba_premultiplied(
                            p.color.r(), p.color.g(), p.color.b(), center_alpha
                        );
                        painter.circle_filled(pos, size * 0.35, center);
                    }
                }
                ParticleShape::Star => {
                    // 3D star with glow
                    let glow_alpha = (alpha / 4).max(3);
                    let glow_color = Color32::from_rgba_premultiplied(
                        p.color.r(), p.color.g(), p.color.b(), glow_alpha
                    );
                    painter.circle_filled(pos, size * 1.5, glow_color);
                    self.draw_star(painter, pos, size, p.angle, color);
                }
                ParticleShape::Diamond => {
                    self.draw_diamond(painter, pos, size, p.angle, color);
                }
                ParticleShape::Triangle => {
                    self.draw_triangle(painter, pos, size, p.angle, color);
                }
                ParticleShape::Spark => {
                    self.draw_spark(painter, pos, size, p.angle, p.vel, color);
                }
            }
        }
    }
    
    fn draw_star(&self, painter: &Painter, center: Pos2, size: f32, angle: f32, color: Color32) {
        let points = 5;
        let outer_r = size;
        let inner_r = size * 0.4;
        
        let mut vertices = Vec::with_capacity(points * 2);
        
        for i in 0..(points * 2) {
            let a = angle + (i as f32 * PI / points as f32);
            let r = if i % 2 == 0 { outer_r } else { inner_r };
            vertices.push(Pos2::new(
                center.x + a.cos() * r,
                center.y + a.sin() * r,
            ));
        }
        
        // Draw as triangles from center
        for i in 0..vertices.len() {
            let next = (i + 1) % vertices.len();
            painter.line_segment([center, vertices[i]], egui::Stroke::new(1.0, color));
            painter.line_segment([vertices[i], vertices[next]], egui::Stroke::new(1.0, color));
        }
    }
    
    fn draw_diamond(&self, painter: &Painter, center: Pos2, size: f32, angle: f32, color: Color32) {
        let points = [
            Pos2::new(center.x + (angle).cos() * size, center.y + (angle).sin() * size),
            Pos2::new(center.x + (angle + PI * 0.5).cos() * size * 0.6, center.y + (angle + PI * 0.5).sin() * size * 0.6),
            Pos2::new(center.x + (angle + PI).cos() * size, center.y + (angle + PI).sin() * size),
            Pos2::new(center.x + (angle + PI * 1.5).cos() * size * 0.6, center.y + (angle + PI * 1.5).sin() * size * 0.6),
        ];
        
        for i in 0..4 {
            painter.line_segment([points[i], points[(i + 1) % 4]], egui::Stroke::new(2.0, color));
        }
    }
    
    fn draw_triangle(&self, painter: &Painter, center: Pos2, size: f32, angle: f32, color: Color32) {
        let points: Vec<Pos2> = (0..3)
            .map(|i| {
                let a = angle + (i as f32 * PI * 2.0 / 3.0);
                Pos2::new(center.x + a.cos() * size, center.y + a.sin() * size)
            })
            .collect();
        
        for i in 0..3 {
            painter.line_segment([points[i], points[(i + 1) % 3]], egui::Stroke::new(2.0, color));
        }
    }
    
    fn draw_spark(&self, painter: &Painter, center: Pos2, size: f32, _angle: f32, vel: Vec2, color: Color32) {
        let vel_len = (vel.x * vel.x + vel.y * vel.y).sqrt().max(0.1);
        let dir = Vec2::new(vel.x / vel_len, vel.y / vel_len);
        
        let length = size * 2.0;
        let end = Pos2::new(center.x - dir.x * length, center.y - dir.y * length);
        
        painter.line_segment([center, end], egui::Stroke::new(2.0, color));
        painter.circle_filled(center, size * 0.5, color);
    }
    
    /// Draw volumetric particle with smooth radial gradient
    /// Creates realistic 3D sphere appearance using Gaussian falloff
    fn draw_volumetric_particle(
        &self, 
        painter: &Painter, 
        pos: Pos2, 
        size: f32, 
        base_color: Color32,
        base_alpha: u8,
        config: &ParticleConfig
    ) {
        let steps = config.volumetric_steps.clamp(8, 48) as usize;
        let intensity = config.glow_intensity.clamp(0.1, 1.0);
        let alpha_f32 = base_alpha as f32 * intensity;
        
        // Draw from outside in for proper layering
        for i in (0..steps).rev() {
            let t = i as f32 / steps as f32;
            
            // Radius goes from size * 1.5 (outer) to size * 0.1 (center)
            let radius = size * (0.1 + t * 1.4);
            
            // Gaussian falloff for smooth gradient: exp(-3 * t^2)
            let gaussian = (-3.0 * t * t).exp();
            
            // Additional brightness boost at center
            let center_boost = if t < 0.3 { 1.0 + (0.3 - t) * 2.0 } else { 1.0 };
            
            let alpha = (alpha_f32 * gaussian * center_boost * 0.7) as u8;
            
            if alpha < 1 { continue; }
            
            // Slight brightening towards center
            let brightness = (1.0 + (1.0 - t) * 0.3).min(1.5);
            let r = ((base_color.r() as f32 * brightness) as u8).min(255);
            let g = ((base_color.g() as f32 * brightness) as u8).min(255);
            let b = ((base_color.b() as f32 * brightness) as u8).min(255);
            
            let layer_color = Color32::from_rgba_premultiplied(r, g, b, alpha);
            painter.circle_filled(pos, radius, layer_color);
        }
        
        // Hot white center for extra pop
        let center_alpha = (alpha_f32 * 0.9) as u8;
        if center_alpha > 10 {
            let r = (base_color.r() as u16 + 50).min(255) as u8;
            let g = (base_color.g() as u16 + 50).min(255) as u8;
            let b = (base_color.b() as u16 + 50).min(255) as u8;
            let center_color = Color32::from_rgba_premultiplied(r, g, b, center_alpha);
            painter.circle_filled(pos, size * 0.15, center_color);
        }
    }
    
    /// Get iterator over particles for external rendering
    pub fn get_particles_iter(&self) -> std::slice::Iter<'_, Particle> {
        self.particles.iter()
    }
    
    /// Placeholder for compatibility
    pub fn clone_render_data(&self) {}
}