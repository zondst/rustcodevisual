//! Particle Engine for Particle Studio RS
//! Advanced particle system with physics, audio reactivity, and multiple modes

use crate::audio::{AudioState, NormalizedAudio};
use crate::config::{
    ColorScheme, ConnectionConfig, DeathSpiralConfig, ParticleConfig, ParticleMode, ParticleShape,
    TrailConfig,
};
use egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
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
    pub audio_alpha: f32, // Opacity driven by audio (0.0-1.0)
    pub audio_size: f32,  // Size multiplier from audio
    // Death Spiral state
    pub ring_index: usize,     // Which ring this particle belongs to
    pub position_in_ring: f32, // Position within the ring (0.0-1.0)
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
            ring_index: 0,
            position_in_ring: 0.0,
        }
    }
}

/// Trail point for particle trail system
#[derive(Clone, Copy)]
pub struct TrailPoint {
    pub pos: Vec2,
    pub age: f32, // 0.0 = new, 1.0 = fully faded
    pub size: f32,
    pub brightness: f32,
    pub color: Color32,
}

/// Trail system for all particles
pub struct TrailSystem {
    /// Trail points for each particle (indexed by particle index)
    pub trails: Vec<Vec<TrailPoint>>,
    pub max_trail_length: usize,
    pub trail_spawn_rate: f32,
    pub trail_fade_speed: f32,
    last_spawn_time: f32,
}

impl TrailSystem {
    pub fn new(particle_count: usize, max_length: usize) -> Self {
        Self {
            trails: vec![Vec::with_capacity(max_length); particle_count],
            max_trail_length: max_length,
            trail_spawn_rate: 30.0,
            trail_fade_speed: 2.0,
            last_spawn_time: 0.0,
        }
    }

    pub fn update(&mut self, particles: &[Particle], config: &TrailConfig, dt: f32) {
        if !config.enabled || config.opacity < 0.01 {
            // Clear trails if disabled
            for trail in &mut self.trails {
                trail.clear();
            }
            return;
        }

        self.trail_spawn_rate = config.spawn_rate;
        self.trail_fade_speed = config.fade_speed;
        self.max_trail_length = config.max_length;

        self.last_spawn_time += dt;
        let spawn_interval = 1.0 / self.trail_spawn_rate;
        let should_spawn = self.last_spawn_time >= spawn_interval;

        if should_spawn {
            self.last_spawn_time = 0.0;
        }

        // Ensure we have enough trail vectors
        if self.trails.len() < particles.len() {
            self.trails
                .resize(particles.len(), Vec::with_capacity(self.max_trail_length));
        }

        for (i, particle) in particles.iter().enumerate() {
            if i >= self.trails.len() {
                continue;
            }

            let trail = &mut self.trails[i];

            // Add new point if spawning and particle is visible
            if should_spawn && particle.audio_alpha > 0.1 {
                // Check for wrap-around (large distance from last point)
                let mut is_wrap = false;
                if let Some(last) = trail.last() {
                    let dx = (particle.pos.x - last.pos.x).abs();
                    let dy = (particle.pos.y - last.pos.y).abs();
                    if dx > 100.0 || dy > 100.0 {
                        is_wrap = true;
                    }
                }

                if is_wrap {
                    trail.clear();
                } else {
                    let point = TrailPoint {
                        pos: particle.pos,
                        age: 0.0,
                        size: particle.size * config.width_scale,
                        brightness: particle.brightness * config.opacity,
                        color: particle.color,
                    };

                    if trail.len() >= self.max_trail_length {
                        trail.remove(0);
                    }
                    trail.push(point);
                }
            }

            // Update age and remove old points
            trail.retain_mut(|point| {
                point.age += dt * self.trail_fade_speed;
                point.brightness *= 0.98;
                point.age < 1.0
            });
        }
    }

    pub fn resize(&mut self, particle_count: usize) {
        self.trails
            .resize(particle_count, Vec::with_capacity(self.max_trail_length));
    }

    pub fn render(&self, painter: &Painter, rect: Rect, config: &TrailConfig, _audio: &AudioState) {
        if !config.enabled {
            return;
        }

        for trail in &self.trails {
            if trail.len() < 2 {
                continue;
            }

            // Draw trail as connected line segments with fading
            for i in 1..trail.len() {
                let p0 = &trail[i - 1];
                let p1 = &trail[i];

                let alpha0 = ((1.0 - p0.age) * p0.brightness * 255.0 * config.opacity) as u8;
                let alpha1 = ((1.0 - p1.age) * p1.brightness * 255.0 * config.opacity) as u8;

                if alpha0 < 2 && alpha1 < 2 {
                    continue;
                }

                let pos0 = rect.min + p0.pos;
                let pos1 = rect.min + p1.pos;

                // Use gradient between the two points
                let avg_alpha = (alpha0 as u16 + alpha1 as u16) / 2;
                let color = Color32::from_rgba_premultiplied(
                    p0.color.r(),
                    p0.color.g(),
                    p0.color.b(),
                    avg_alpha as u8,
                );

                let thickness = (p0.size + p1.size) / 2.0 * (1.0 - (p0.age + p1.age) / 2.0);
                painter.line_segment([pos0, pos1], Stroke::new(thickness.max(0.5), color));

                // Add glow effect if enabled
                if config.glow_enabled && avg_alpha > 10 {
                    let glow_alpha = (avg_alpha as f32 * 0.3) as u8;
                    let glow_color = Color32::from_rgba_premultiplied(
                        p0.color.r(),
                        p0.color.g(),
                        p0.color.b(),
                        glow_alpha,
                    );
                    painter.line_segment([pos0, pos1], Stroke::new(thickness * 2.0, glow_color));
                }
            }
        }
    }
}

/// Connection between two particles
pub struct Connection {
    pub particle_a: usize,
    pub particle_b: usize,
    pub strength: f32,
}

pub struct SpatialGrid {
    cell_size: f32,
    cells: HashMap<(i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            cells: HashMap::new(),
        }
    }

    pub fn clear(&mut self) {
        for cells in self.cells.values_mut() {
            cells.clear();
        }
    }

    pub fn insert(&mut self, index: usize, pos: Vec2) {
        let cell = (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
        );
        self.cells.entry(cell).or_default().push(index);
    }

    pub fn query_radius(&self, pos: Vec2, radius: f32) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let cell_x = (pos.x / self.cell_size).floor() as i32;
        let cell_y = (pos.y / self.cell_size).floor() as i32;
        let search_range = (radius / self.cell_size).ceil() as i32;

        for dx in -search_range..=search_range {
            for dy in -search_range..=search_range {
                if let Some(indices) = self.cells.get(&(cell_x + dx, cell_y + dy)) {
                    neighbors.extend_from_slice(indices);
                }
            }
        }
        neighbors
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

    // Trail system
    pub trail_system: TrailSystem,

    // Spatial Grid for optimizations
    spatial_grid: SpatialGrid,

    // Connections cache
    pub connections: Vec<Connection>,

    // Death Spiral state
    death_spiral_initialized: bool,
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
            trail_system: TrailSystem::new(10000, 20),
            spatial_grid: SpatialGrid::new(40.0),
            connections: Vec::new(),
            death_spiral_initialized: false,
        }
    }

    pub fn update_palette(&mut self, colors: &ColorScheme) {
        self.palette = colors
            .particles
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

    pub fn update(
        &mut self,
        config: &ParticleConfig,
        connection_config: &ConnectionConfig,
        death_spiral_config: Option<&DeathSpiralConfig>,
        audio: &AudioState,
        dt: f32,
        normalized: Option<&NormalizedAudio>,
    ) {
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

        // Apply mode-specific forces
        // Apply mode-specific forces
        if config.mode == ParticleMode::Chaos {
            self.update_chaos(config, audio, dt, speed_factor, normalized);
        } else if config.mode == ParticleMode::Calm {
            self.update_calm(config, audio, dt, speed_factor, normalized);
        } else if config.mode == ParticleMode::Cinematic {
            self.update_cinematic(config, audio, dt, speed_factor, normalized);
        } else if config.mode == ParticleMode::Orbit {
            self.update_orbit(config, audio, dt, speed_factor, normalized);
        } else if config.mode == ParticleMode::DeathSpiral {
            if let Some(spiral_config) = death_spiral_config {
                self.update_death_spiral(
                    config,
                    spiral_config,
                    audio,
                    dt,
                    speed_factor,
                    normalized,
                );
            }
        }

        // Calculate connections once per frame
        // Grid Maintenance
        // We must rebuild grid if EITHER connections OR repulsion is enabled to avoid stale indices panic
        let mut grid_cell_size: f32 = 0.0;
        if connection_config.enabled {
            grid_cell_size = grid_cell_size.max(connection_config.max_distance);
        }
        if config.repulsion_enabled {
            grid_cell_size = grid_cell_size.max(config.repulsion_radius);
        }

        if grid_cell_size > 0.0 {
            // Update grid settings if needed (allow some tolerance to avoid thrashing)
            if (self.spatial_grid.cell_size - grid_cell_size).abs() > 5.0 {
                self.spatial_grid = SpatialGrid::new(grid_cell_size);
            }

            // Critical: Rebuild grid with CURRENT particles prevents panic
            self.spatial_grid.clear();
            for (i, p) in self.particles.iter().enumerate() {
                self.spatial_grid.insert(i, p.pos);
            }
        }

        // Calculate connections once per frame
        if connection_config.enabled {
            self.connections = self.find_connections(connection_config, audio);
        } else {
            self.connections.clear();
        }

        // ========================================================
        // RULE 3: Update all particles with audio-physics
        // ========================================================
        let width = self.width;
        let height = self.height;
        let repulsion_enabled = config.repulsion_enabled;
        let repulsion_strength = config.repulsion_strength;
        let repulsion_radius = config.repulsion_radius;
        let _spatial_for_repulsion = if repulsion_enabled {
            Some(&self.spatial_grid)
        } else {
            None
        };

        // We can't use parallel iteration easily if we need random access for repulsion via grid
        // So we'll iterate sequentially or use a two-pass approach
        // For now, sequential update for repulsion simplicity

        let particle_count = self.particles.len();

        // Create a copy of positions for repulsion query to avoid borrowing issues
        let positions: Vec<Vec2> = if repulsion_enabled {
            self.particles.iter().map(|p| p.pos).collect()
        } else {
            Vec::new()
        };

        for i in 0..particle_count {
            // -- REPULSION --
            if repulsion_enabled {
                let p_pos = positions[i];
                let neighbors = self.spatial_grid.query_radius(p_pos, repulsion_radius);
                let mut force = Vec2::ZERO;

                for &j in &neighbors {
                    if i == j {
                        continue;
                    }
                    let other_pos = positions[j];
                    let dx = p_pos.x - other_pos.x;
                    let dy = p_pos.y - other_pos.y;
                    let dist_sq = dx * dx + dy * dy;

                    if dist_sq < repulsion_radius * repulsion_radius && dist_sq > 0.1 {
                        let dist = dist_sq.sqrt();
                        let f = (1.0 - dist / repulsion_radius) * repulsion_strength;
                        force.x += (dx / dist) * f;
                        force.y += (dy / dist) * f;
                    }
                }

                // Check bounds to avoid panic
                if let Some(p) = self.particles.get_mut(i) {
                    p.vel += force * dt * 60.0;
                }
            }

            let p = &mut self.particles[i];
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
                    0.0 // Fade to invisible when no audio
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
                p.audio_alpha = 1.0; // Always visible in non-reactive mode
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
            if p.pos.x < -50.0 {
                p.pos.x = width + 50.0;
            }
            if p.pos.x > width + 50.0 {
                p.pos.x = -50.0;
            }
            if p.pos.y < -50.0 {
                p.pos.y = height + 50.0;
            }
            if p.pos.y > height + 50.0 {
                p.pos.y = -50.0;
            }
        }

        // ========================================================
        // RULE 4: Remove particles when fully faded or dead
        // ========================================================
        if config.audio_reactive_spawn {
            self.particles
                .retain(|p| p.audio_alpha > 0.01 && p.life > 0.0);
        } else {
            // Non-reactive mode: reset dead particles
            let dead_indices: Vec<usize> = self
                .particles
                .iter()
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
    fn spawn_audio_particle(
        &mut self,
        config: &ParticleConfig,
        audio: &AudioState,
        rng: &mut impl Rng,
        center: Vec2,
    ) {
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
            audio_alpha: 0.1, // Start low, will ramp up
            audio_size: audio.smooth_amplitude.clamp(0.3, 1.0),
            ring_index: 0,
            position_in_ring: 0.0,
        };

        self.particles.push(particle);
    }

    fn update_chaos(
        &mut self,
        config: &ParticleConfig,
        audio: &AudioState,
        _dt: f32,
        speed_factor: f32,
        normalized: Option<&NormalizedAudio>,
    ) {
        let center = Vec2::new(self.width / 2.0, self.height / 2.0);
        let flow_field_time = self.flow_field_time;

        // Resolve effective audio values
        let (eff_mid, eff_beat) = if let Some(norm) = normalized {
            (norm.mid, if norm.has_bass_hit { 1.0 } else { 0.0 })
        } else {
            (audio.mid, audio.beat)
        };

        self.particles.par_iter_mut().for_each(|p| {
            // Curl noise flow field
            let noise_scale = 0.003;
            let nx = p.pos.x * noise_scale + flow_field_time;
            let ny = p.pos.y * noise_scale + flow_field_time * 0.7;

            let flow_x = (ny * 6.0).sin() + (nx * 3.0).cos() * 0.5;
            let flow_y = (nx * 6.0).cos() + (ny * 3.0).sin() * 0.5;

            p.vel.x += flow_x * 0.1 * speed_factor * (1.0 + eff_mid);
            p.vel.y += flow_y * 0.1 * speed_factor * (1.0 + eff_mid);

            // Beat explosion
            if eff_beat > 0.5 {
                let dx = p.pos.x - center.x;
                let dy = p.pos.y - center.y;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);

                let force = eff_beat * config.speed * 2.0;
                p.vel.x += (dx / dist) * force;
                p.vel.y += (dy / dist) * force;
            }

            // Damping
            p.vel *= 0.98;

            // Gravity
            p.vel.y += config.gravity * speed_factor * 0.5;
        });
    }

    fn update_calm(
        &mut self,
        _config: &ParticleConfig,
        audio: &AudioState,
        _dt: f32,
        speed_factor: f32,
        normalized: Option<&NormalizedAudio>,
    ) {
        let time = self.time;
        // Resolve effective audio
        let eff_amp = normalized.map(|n| n.intensity).unwrap_or(audio.amplitude);

        self.particles.par_iter_mut().for_each(|p| {
            // Gentle floating motion
            let float_x = (time * 0.5 + p.pos.y * 0.01).sin() * 0.02;
            let float_y = -0.02 - eff_amp * 0.03;

            p.vel.x += float_x * speed_factor;
            p.vel.y += float_y * speed_factor;

            // Strong damping for calm effect
            p.vel *= 0.96;

            // Gravity (upward drift in calm mode)
            p.vel.y -= 0.01 * speed_factor;
        });
    }

    fn update_cinematic(
        &mut self,
        config: &ParticleConfig,
        audio: &AudioState,
        dt: f32,
        speed_factor: f32,
        normalized: Option<&NormalizedAudio>,
    ) {
        let center = Vec2::new(self.width / 2.0, self.height / 2.0);
        let time = self.time;
        // Resolve effective audio
        let eff_beat = normalized
            .map(|n| if n.has_bass_hit { 1.0 } else { 0.0 })
            .unwrap_or(audio.beat);

        self.particles.par_iter_mut().for_each(|p| {
            let dx = p.pos.x - center.x;
            let dy = p.pos.y - center.y;
            let dist = (dx * dx + dy * dy).sqrt().max(1.0);

            // Audio-driven speed control
            let audio_energy = normalized
                .map(|n| n.intensity)
                .unwrap_or(audio.smooth_amplitude)
                .clamp(0.0, 1.0);
            let dynamic_speed = config.speed * (0.2 + 0.8 * audio_energy);

            // Slow orbital motion
            let angle = dist * 0.005 + time * 0.1;
            let target_vel_x = -angle.sin() * dynamic_speed * 10.0;
            let target_vel_y = angle.cos() * dynamic_speed * 10.0;

            // Smoothly interpolate velocity
            let lerp_factor = 2.0 * dt;
            p.vel.x += (target_vel_x - p.vel.x) * lerp_factor;
            p.vel.y += (target_vel_y - p.vel.y) * lerp_factor;

            // Gentle push out on beat
            if eff_beat > 0.2 {
                let push = eff_beat * 0.5 * speed_factor;
                p.vel.x += (dx / dist) * push;
                p.vel.y += (dy / dist) * push;
            }

            // Strong damping, stronger on silence to stop "floating aimlessly"
            let damping = if audio_energy < 0.05 { 0.90 } else { 0.99 };
            p.vel *= damping;
        });
    }

    fn update_orbit(
        &mut self,
        config: &ParticleConfig,
        audio: &AudioState,
        _dt: f32,
        speed_factor: f32,
        normalized: Option<&NormalizedAudio>,
    ) {
        let center = Vec2::new(self.width / 2.0, self.height / 2.0);

        // Resolve effective audio
        let eff_beat = normalized
            .map(|n| if n.has_bass_hit { 1.0 } else { 0.0 })
            .unwrap_or(audio.beat);
        let audio_energy = normalized
            .map(|n| n.intensity)
            .unwrap_or(audio.smooth_amplitude)
            .clamp(0.0, 1.0);

        self.particles.par_iter_mut().for_each(|p| {
            let dx = p.pos.x - center.x;
            let dy = p.pos.y - center.y;
            let dist = (dx * dx + dy * dy).sqrt().max(1.0);

            // Perpendicular velocity (orbit)
            let perp_x = -dy / dist;
            let perp_y = dx / dist;

            let orbit_speed =
                config.orbit_speed * (0.3 + 0.7 * audio_energy) * (1.0 + eff_beat * 0.5);
            p.vel.x += perp_x * orbit_speed * speed_factor;
            p.vel.y += perp_y * orbit_speed * speed_factor;

            // Pull toward center
            let pull_strength = 0.02;
            p.vel.x -= (dx / dist) * pull_strength * speed_factor;
            p.vel.y -= (dy / dist) * pull_strength * speed_factor;

            // Moderate damping, brake on silence
            let damping = if audio_energy < 0.05 { 0.92 } else { 0.97 };
            p.vel *= damping;
        });
    }

    /// Update particles in Death Spiral mode (ant mill / death dance effect)
    /// Creates hypnotic concentric rings of particles following each other
    pub fn update_death_spiral(
        &mut self,
        config: &ParticleConfig,
        spiral_config: &DeathSpiralConfig,
        audio: &AudioState,
        dt: f32,
        speed_factor: f32,
        normalized: Option<&NormalizedAudio>,
    ) {
        let center = Vec2::new(self.width / 2.0, self.height / 2.0);
        let time = self.time;

        // Resolve effective audio
        let (eff_bass, eff_amp, eff_beat) = if let Some(norm) = normalized {
            // Map normalized 0-1 range to similar scale as raw audio might trigger
            (
                norm.bass,
                norm.intensity,
                if norm.has_bass_hit { 0.8 } else { 0.0 },
            )
        } else {
            (audio.smooth_bass, audio.smooth_amplitude, audio.smooth_beat)
        };

        // Initialize particles into rings if needed
        if !self.death_spiral_initialized || self.particles.is_empty() {
            self.initialize_death_spiral(config, spiral_config);
            self.death_spiral_initialized = true;
        }

        // Audio-reactive parameters
        let audio_speed_mult = 1.0 + eff_bass * spiral_config.audio_speed_influence;
        let audio_radius_mult = 1.0 + eff_amp * spiral_config.audio_radius_influence;
        let beat_pulse = 1.0 + eff_beat * spiral_config.beat_pulse_strength;

        let ring_count = spiral_config.ring_count.max(1);
        let particles_total = self.particles.len();
        let particles_per_ring = particles_total / ring_count.max(1);

        // Use indices for parallel iteration
        let width = self.width;
        let height = self.height;

        self.particles
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, p)| {
                // Determine ring and position in ring
                let ring_idx = if particles_per_ring > 0 {
                    (idx / particles_per_ring).min(ring_count - 1)
                } else {
                    0
                };
                let position_in_ring = if particles_per_ring > 0 {
                    (idx % particles_per_ring) as f32 / particles_per_ring.max(1) as f32
                } else {
                    idx as f32 / particles_total.max(1) as f32
                };

                // Store ring info on particle
                // Note: We can't modify these in parallel, so we use computed values

                // Base radius for this ring (exponential spacing)
                let base_radius = spiral_config.inner_radius
                    * spiral_config.ring_spacing.powf(ring_idx as f32)
                    * audio_radius_mult;

                // Scale radius to fit screen
                let max_radius = (width.min(height) / 2.0) * 0.85;
                let scaled_radius = base_radius.min(max_radius);

                // Direction: alternate for visual effect
                let direction = if spiral_config.alternate_direction && ring_idx % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };

                // Speed: inner rings spin faster
                let ring_speed = spiral_config.base_rotation_speed
                    * (1.0 + 0.3 * (ring_count - ring_idx) as f32)
                    * audio_speed_mult
                    * direction;

                // Base angle + rotation from time
                let base_angle = position_in_ring * PI * 2.0;
                let rotated_angle = base_angle + time * ring_speed;

                // Wave distortion for organic breathing effect
                let wave_offset = (rotated_angle * spiral_config.wave_frequency + time * 2.0).sin()
                    * spiral_config.wave_amplitude
                    * scaled_radius;

                // Spiral tightness (inward/outward spiral motion)
                let spiral_offset =
                    position_in_ring * spiral_config.spiral_tightness * scaled_radius * 0.5;

                // Final radius with beat pulse
                let final_radius = (scaled_radius + wave_offset + spiral_offset) * beat_pulse;

                // Target position
                let target_x = center.x + rotated_angle.cos() * final_radius;
                let target_y = center.y + rotated_angle.sin() * final_radius;
                let target = Vec2::new(target_x, target_y);

                // Follow behavior - particle moves toward target with inertia
                let to_target = target - p.pos;
                let follow_force = spiral_config.follow_strength * (1.0 + audio.smooth_mid * 0.5);

                p.vel.x += to_target.x * follow_force * speed_factor;
                p.vel.y += to_target.y * follow_force * speed_factor;

                // Strong damping for smooth motion
                p.vel *= 0.92;

                // Clamp velocity to prevent chaos
                let vel_mag = (p.vel.x * p.vel.x + p.vel.y * p.vel.y).sqrt();
                let max_vel = 8.0 * audio_speed_mult;
                if vel_mag > max_vel {
                    let scale = max_vel / vel_mag;
                    p.vel.x *= scale;
                    p.vel.y *= scale;
                }

                // Angular velocity synced with ring
                p.angular_vel = ring_speed * 0.5;

                // Size pulses with beat, outer rings slightly larger
                let ring_size_mult = 1.0 + ring_idx as f32 * 0.1;
                p.size = p.base_size * beat_pulse * ring_size_mult;

                // Brightness based on audio
                p.brightness = 0.7 + audio.smooth_amplitude * 0.3 + audio.smooth_beat * 0.2;

                // Keep audio_alpha high for visibility
                if p.audio_alpha < 0.8 {
                    p.audio_alpha += dt * 2.0;
                }
                p.audio_alpha = p.audio_alpha.min(1.0);
            });
    }

    /// Initialize particles into death spiral formation
    fn initialize_death_spiral(
        &mut self,
        config: &ParticleConfig,
        spiral_config: &DeathSpiralConfig,
    ) {
        let center = Vec2::new(self.width / 2.0, self.height / 2.0);
        let ring_count = spiral_config.ring_count.max(1);

        // Ensure we have enough particles
        let target_count = config
            .count
            .max(ring_count * spiral_config.particles_per_ring);

        let mut rng = rand::thread_rng();

        // Clear and respawn with ring assignments
        self.particles.clear();

        for ring_idx in 0..ring_count {
            let particles_in_ring = if ring_idx < ring_count - 1 {
                spiral_config.particles_per_ring
            } else {
                // Last ring gets remaining particles
                target_count - self.particles.len()
            };

            let base_radius =
                spiral_config.inner_radius * spiral_config.ring_spacing.powf(ring_idx as f32);

            let max_radius = (self.width.min(self.height) / 2.0) * 0.85;
            let scaled_radius = base_radius.min(max_radius);

            for i in 0..particles_in_ring {
                let position_in_ring = i as f32 / particles_in_ring as f32;
                let angle = position_in_ring * PI * 2.0;

                let pos = Vec2::new(
                    center.x + angle.cos() * scaled_radius,
                    center.y + angle.sin() * scaled_radius,
                );

                let color_idx = rng.gen_range(0..self.palette.len());
                let base_size =
                    config.min_size + rng.gen::<f32>() * (config.max_size - config.min_size);

                let particle = Particle {
                    pos,
                    vel: Vec2::ZERO,
                    life: 10.0, // Long life for death spiral
                    max_life: 10.0,
                    size: base_size,
                    base_size,
                    color: self.palette[color_idx],
                    color_index: color_idx,
                    angle: angle,
                    angular_vel: 0.0,
                    brightness: 1.0,
                    audio_alpha: 1.0,
                    audio_size: 1.0,
                    ring_index: ring_idx,
                    position_in_ring,
                };

                self.particles.push(particle);
            }
        }
    }

    /// Reset death spiral when mode changes
    pub fn reset_death_spiral(&mut self) {
        self.death_spiral_initialized = false;
    }

    pub fn find_connections(
        &mut self,
        config: &ConnectionConfig,
        audio: &AudioState,
    ) -> Vec<Connection> {
        if !config.enabled {
            return Vec::new();
        }

        let mut connections = Vec::new();

        // Effective max distance based on strict audio reactivity
        // If audio_reactive is ON, distance scales with volume. Silence = no connections.
        let audio_scale = if config.audio_reactive {
            // Map 0.0-1.0 amplitude to 0.1-1.2 scale (keep minimum 10% distance to avoid total collapse, or 0? User asked for strict)
            // User said "completely not under music... even when turned off". So strict 0 at 0 is better.
            audio.smooth_amplitude.clamp(0.0, 1.0) * 1.5
        } else {
            1.0
        };

        // Don't calculate if scale is effectively zero (silence)
        if config.audio_reactive && audio_scale < 0.05 {
            return Vec::new();
        }

        let effective_max_dist = config.max_distance * audio_scale;
        let max_dist_sq = effective_max_dist * effective_max_dist;

        // Note: Grid is now built in update() to serve both Connections and Repulsion safely.

        // Track density to limit connections per area
        let mut connection_density: HashMap<(i32, i32), f32> = HashMap::new();
        let density_cell_size = 20.0; // Fixed size for density check

        // Find connections
        for (i, p) in self.particles.iter().enumerate() {
            if p.audio_alpha < config.min_particle_alpha {
                continue;
            }

            // Query using the BASE config distance (grid was built with this in mind)
            let neighbors = self.spatial_grid.query_radius(p.pos, config.max_distance);
            let mut particle_connections = 0;

            for &j in &neighbors {
                if j <= i || particle_connections >= config.max_connections {
                    continue;
                }

                let p2 = &self.particles[j];

                // IMPORTANT: Since grid now contains ALL particles (for repulsion),
                // we must manually skip invisible particles for connections
                if p2.audio_alpha < config.min_particle_alpha {
                    continue;
                }

                let dx = p.pos.x - p2.pos.x;
                let dy = p.pos.y - p2.pos.y;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < max_dist_sq {
                    // Density check
                    let cell = (
                        (p.pos.x / density_cell_size) as i32,
                        (p.pos.y / density_cell_size) as i32,
                    );
                    let cell_density = connection_density.get(&cell).copied().unwrap_or(0.0);

                    if cell_density > config.density_limit {
                        continue;
                    }

                    // Strength relative to the EFFECTIVE distance
                    let strength = 1.0 - (dist_sq.sqrt() / effective_max_dist);

                    connections.push(Connection {
                        particle_a: i,
                        particle_b: j,
                        strength,
                    });

                    *connection_density.entry(cell).or_insert(0.0) += 1.0;
                    particle_connections += 1;
                }
            }
        }

        connections
    }

    /// Render connections between particles (using cached)
    pub fn render_connections(
        &self,
        painter: &Painter,
        rect: Rect,
        config: &ConnectionConfig,
        audio: &AudioState,
    ) {
        if !config.enabled {
            return;
        }

        // Strict audio reactivity for opacity too
        let audio_mult = if config.audio_reactive {
            // Silence = 0 opacity. Max volume = 1.0 (or slightly boosted)
            audio.smooth_amplitude.clamp(0.0, 1.0) * 1.2
        } else {
            1.0
        };

        if config.audio_reactive && audio_mult < 0.05 {
            return;
        }

        for conn in &self.connections {
            if conn.particle_a >= self.particles.len() || conn.particle_b >= self.particles.len() {
                continue;
            }

            let p_a = &self.particles[conn.particle_a];
            let p_b = &self.particles[conn.particle_b];

            let pos_a = rect.min + p_a.pos;
            let pos_b = rect.min + p_b.pos;

            // SAFETY CHECK: If wrap occurred, distance will be large. Don't draw line across screen.
            let dx = (p_a.pos.x - p_b.pos.x).abs();
            let dy = (p_a.pos.y - p_b.pos.y).abs();
            if dx > config.max_distance * 1.5 || dy > config.max_distance * 1.5 {
                continue;
            }

            let alpha = (conn.strength * config.opacity * audio_mult * 255.0) as u8;
            if alpha < 2 {
                continue;
            }

            // Gradient between particle colors
            let color = if config.gradient_enabled {
                let r = ((p_a.color.r() as u16 + p_b.color.r() as u16) / 2) as u8;
                let g = ((p_a.color.g() as u16 + p_b.color.g() as u16) / 2) as u8;
                let b = ((p_a.color.b() as u16 + p_b.color.b() as u16) / 2) as u8;
                Color32::from_rgba_premultiplied(r, g, b, alpha)
            } else {
                Color32::from_rgba_premultiplied(p_a.color.r(), p_a.color.g(), p_a.color.b(), alpha)
            };

            let thickness = if config.fade_by_distance {
                config.thickness * conn.strength * audio_mult
            } else {
                config.thickness * audio_mult
            };

            painter.line_segment([pos_a, pos_b], Stroke::new(thickness.max(0.5), color));
        }
    }

    /// Update trail system
    pub fn update_trails(&mut self, config: &TrailConfig, dt: f32) {
        self.trail_system.update(&self.particles, config, dt);
    }

    /// Render trails
    pub fn render_trails(
        &self,
        painter: &Painter,
        rect: Rect,
        config: &TrailConfig,
        audio: &AudioState,
    ) {
        self.trail_system.render(painter, rect, config, audio);
    }

    fn spawn_particle(&mut self, config: &ParticleConfig, rng: &mut rand::rngs::ThreadRng) {
        let mut p = Particle::default();
        self.reset_particle(&mut p, config, rng);
        self.particles.push(p);
    }

    fn reset_particle(
        &self,
        p: &mut Particle,
        config: &ParticleConfig,
        rng: &mut rand::rngs::ThreadRng,
    ) {
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

    fn reset_particle_at(
        &mut self,
        idx: usize,
        config: &ParticleConfig,
        rng: &mut rand::rngs::ThreadRng,
    ) {
        if idx >= self.particles.len() {
            return;
        }

        let width = self.width;
        let height = self.height;
        let palette_len = self.palette.len();
        let palette = self.palette.clone();

        let p = &mut self.particles[idx];

        // Random position
        p.pos = Vec2::new(rng.gen_range(0.0..width), rng.gen_range(0.0..height));

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
    pub fn render(
        &self,
        painter: &Painter,
        rect: Rect,
        config: &ParticleConfig,
        audio: &AudioState,
        normalized: Option<&NormalizedAudio>,
    ) {
        if !config.enabled {
            return;
        }

        // Resolve effective amplitude for global size check
        let eff_amp = if config.adaptive_audio_enabled {
            normalized.map(|n| n.intensity).unwrap_or(audio.amplitude)
        } else {
            audio.amplitude
        };

        // Depth Sort: Create indices and sort by brightness/alpha if enabled
        let mut indices: Vec<usize> = (0..self.particles.len()).collect();
        if config.depth_sort_enabled {
            indices.sort_by(|&a, &b| {
                let pa = &self.particles[a];
                let pb = &self.particles[b];
                // Sort by brightness (visual importance)
                let val_a = pa.audio_alpha * pa.brightness * pa.size;
                let val_b = pb.audio_alpha * pb.brightness * pb.size;
                val_a
                    .partial_cmp(&val_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        for &i in &indices {
            let p = &self.particles[i];
            // Use audio_alpha for audio-reactive visibility
            let life_alpha = (p.life / p.max_life).clamp(0.0, 1.0);
            let audio_factor = if config.audio_reactive_spawn {
                p.audio_alpha
            } else {
                1.0
            };
            let alpha = (life_alpha * p.brightness * audio_factor * 255.0).min(255.0) as u8;

            let pos = rect.min + Vec2::new(p.pos.x, p.pos.y);

            // Skip nearly invisible particles
            if alpha < 3 {
                continue;
            }

            let color =
                Color32::from_rgba_premultiplied(p.color.r(), p.color.g(), p.color.b(), alpha);

            let size = p.size * (1.0 + eff_amp * 0.2);

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
                            p.color.r(),
                            p.color.g(),
                            p.color.b(),
                            glow_alpha,
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
                        self.draw_volumetric_particle(
                            painter,
                            pos,
                            size * 1.5,
                            p.color,
                            alpha,
                            config,
                        );
                    } else {
                        // Old layer-based glow (legacy)
                        let intensity = config.glow_intensity.clamp(0.1, 1.0);
                        let base_alpha = (alpha as f32 * intensity) as u8;

                        // Layer 1: Outer soft glow
                        let outer_alpha = (base_alpha / 10).max(2);
                        let outer_glow = Color32::from_rgba_premultiplied(
                            p.color.r(),
                            p.color.g(),
                            p.color.b(),
                            outer_alpha,
                        );
                        painter.circle_filled(pos, size * 1.6, outer_glow);

                        // Layer 2: Mid glow
                        let mid_alpha = (base_alpha / 6).max(3);
                        let mid_glow = Color32::from_rgba_premultiplied(
                            p.color.r(),
                            p.color.g(),
                            p.color.b(),
                            mid_alpha,
                        );
                        painter.circle_filled(pos, size * 1.2, mid_glow);

                        // Layer 3: Core
                        let core_alpha = (base_alpha / 2).max(10);
                        let core = Color32::from_rgba_premultiplied(
                            p.color.r(),
                            p.color.g(),
                            p.color.b(),
                            core_alpha,
                        );
                        painter.circle_filled(pos, size * 0.7, core);

                        // Layer 4: Bright center
                        let center_alpha = base_alpha.min(150);
                        let center = Color32::from_rgba_premultiplied(
                            p.color.r(),
                            p.color.g(),
                            p.color.b(),
                            center_alpha,
                        );
                        painter.circle_filled(pos, size * 0.35, center);
                    }
                }
                ParticleShape::Star => {
                    // 3D star with glow
                    let glow_alpha = (alpha / 4).max(3);
                    let glow_color = Color32::from_rgba_premultiplied(
                        p.color.r(),
                        p.color.g(),
                        p.color.b(),
                        glow_alpha,
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
            vertices.push(Pos2::new(center.x + a.cos() * r, center.y + a.sin() * r));
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
            Pos2::new(
                center.x + (angle).cos() * size,
                center.y + (angle).sin() * size,
            ),
            Pos2::new(
                center.x + (angle + PI * 0.5).cos() * size * 0.6,
                center.y + (angle + PI * 0.5).sin() * size * 0.6,
            ),
            Pos2::new(
                center.x + (angle + PI).cos() * size,
                center.y + (angle + PI).sin() * size,
            ),
            Pos2::new(
                center.x + (angle + PI * 1.5).cos() * size * 0.6,
                center.y + (angle + PI * 1.5).sin() * size * 0.6,
            ),
        ];

        for i in 0..4 {
            painter.line_segment(
                [points[i], points[(i + 1) % 4]],
                egui::Stroke::new(2.0, color),
            );
        }
    }

    fn draw_triangle(
        &self,
        painter: &Painter,
        center: Pos2,
        size: f32,
        angle: f32,
        color: Color32,
    ) {
        let points: Vec<Pos2> = (0..3)
            .map(|i| {
                let a = angle + (i as f32 * PI * 2.0 / 3.0);
                Pos2::new(center.x + a.cos() * size, center.y + a.sin() * size)
            })
            .collect();

        for i in 0..3 {
            painter.line_segment(
                [points[i], points[(i + 1) % 3]],
                egui::Stroke::new(2.0, color),
            );
        }
    }

    fn draw_spark(
        &self,
        painter: &Painter,
        center: Pos2,
        size: f32,
        _angle: f32,
        vel: Vec2,
        color: Color32,
    ) {
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
        config: &ParticleConfig,
    ) {
        let base_radius = size;
        let glow_radius = base_radius * (1.0 + config.glow_intensity * 2.0);
        let steps = config.volumetric_steps.clamp(8, 24) as usize; // Reduced steps for optimization

        // Proper gaussian falloff bloom
        for i in (0..steps).rev() {
            let t = i as f32 / steps as f32; // 0.0 to 1.0 (center to edge)
            let radius = base_radius + (glow_radius - base_radius) * t;

            // Normalized distance for gaussian calculation
            let dist_norm = t;

            // Gaussian: exp(-k * x^2)
            let falloff = (-2.5 * dist_norm * dist_norm).exp();

            // Intensity boost at core
            let intensity = if t < 0.2 { 1.5 } else { 1.0 };

            let alpha_f = base_alpha as f32 * falloff * intensity * (1.0 / steps as f32) * 2.0;
            let alpha = alpha_f.min(255.0) as u8;

            if alpha < 1 {
                continue;
            }

            let color = Color32::from_rgba_premultiplied(
                base_color.r(),
                base_color.g(),
                base_color.b(),
                alpha,
            );

            painter.circle_filled(pos, radius, color);
        }

        // Hot core
        let core_color = Color32::from_rgba_premultiplied(255, 255, 255, 200);
        painter.circle_filled(pos, size * 0.2, core_color);
    }

    /// Get iterator over particles for external rendering
    pub fn get_particles_iter(&self) -> std::slice::Iter<'_, Particle> {
        self.particles.iter()
    }

    /// Placeholder for compatibility
    pub fn clone_render_data(&self) {}
}
