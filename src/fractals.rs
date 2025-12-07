//! Fractal Renderer for Particle Studio RS
//! CPU-based raymarching implementation inspired by fractal_sugar
//! Supports 7 fractal types with audio-reactive coloring

use std::f32::consts::PI;
use image::{ImageBuffer, Rgb, RgbImage};
use rayon::prelude::*;

use crate::presets::FractalType;
use crate::audio::AudioState;

/// 3D Vector for fractal calculations
#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    
    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }
    
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    
    pub fn length_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len < 0.0001 {
            Self::new(1.0, 0.0, 0.0)
        } else {
            Self::new(self.x / len, self.y / len, self.z / len)
        }
    }
    
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    
    pub fn scale(&self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
    
    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
    
    pub fn sub(&self, other: &Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
    
    pub fn min_components(&self, other: &Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }
    
    pub fn abs(&self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
}

/// Quaternion for camera rotation
#[derive(Clone, Copy, Debug)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quaternion {
    pub fn identity() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    }
    
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let half_angle = angle / 2.0;
        let s = half_angle.sin();
        let axis = axis.normalize();
        Self {
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
            w: half_angle.cos(),
        }
    }
    
    pub fn multiply(&self, other: &Self) -> Self {
        Self {
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        }
    }
    
    pub fn rotate_vector(&self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let uv = qv.cross(&v);
        let uuv = qv.cross(&uv);
        
        v.add(&uv.scale(2.0 * self.w)).add(&uuv.scale(2.0))
    }
}

/// Orbit trap for coloring
#[derive(Clone, Copy, Debug)]
pub struct OrbitTrap {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl OrbitTrap {
    pub fn new() -> Self {
        Self { r: 1.0, g: 1.0, b: 1.0 }
    }
    
    pub fn update_min(&mut self, distance: f32, channel: usize) {
        match channel {
            0 => self.r = self.r.min(distance),
            1 => self.g = self.g.min(distance),
            2 => self.b = self.b.min(distance),
            _ => {}
        }
    }
}

/// Fractal renderer configuration
#[derive(Clone, Debug)]
pub struct FractalConfig {
    pub fractal_type: FractalType,
    pub max_iterations: u32,
    pub max_ray_steps: u32,
    pub max_distance: f32,
    pub epsilon: f32,
    pub time: f32,
    pub reactive_bass: Vec3,
    pub reactive_mids: Vec3,
    pub reactive_high: Vec3,
    pub camera_quaternion: Quaternion,
    pub orbit_distance: f32,
    pub kaleidoscope: f32,
}

impl Default for FractalConfig {
    fn default() -> Self {
        Self {
            fractal_type: FractalType::None,
            max_iterations: 100,
            max_ray_steps: 128,
            max_distance: 32.0,
            epsilon: 0.00005,
            time: 0.0,
            reactive_bass: Vec3::zero(),
            reactive_mids: Vec3::zero(),
            reactive_high: Vec3::zero(),
            camera_quaternion: Quaternion::identity(),
            orbit_distance: 2.5,
            kaleidoscope: 0.0,
        }
    }
}

/// Fractal renderer
pub struct FractalRenderer {
    config: FractalConfig,
    width: u32,
    height: u32,
}

impl FractalRenderer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            config: FractalConfig::default(),
            width,
            height,
        }
    }
    
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
    
    pub fn update(&mut self, dt: f32, audio: &AudioState, fractal_type: FractalType) {
        self.config.time += dt;
        self.config.fractal_type = fractal_type;
        
        // Update audio-reactive vectors
        self.config.reactive_bass = Vec3::new(
            audio.bass * 2.0 - 1.0,
            (audio.bass * PI).sin(),
            (audio.bass * PI * 0.5).cos(),
        );
        self.config.reactive_mids = Vec3::new(
            (audio.mid * PI * 1.5).sin(),
            audio.mid * 2.0 - 1.0,
            (audio.mid * PI).cos(),
        );
        self.config.reactive_high = Vec3::new(
            (audio.high * PI * 0.7).cos(),
            (audio.high * PI * 1.3).sin(),
            audio.high * 2.0 - 1.0,
        );
        
        // Rotate camera based on audio
        let rotation_speed = 0.02 + audio.beat * 0.05;
        let axis = Vec3::new(
            (self.config.time * 0.3).sin() * 0.5,
            1.0,
            (self.config.time * 0.2).cos() * 0.3,
        );
        let rotation = Quaternion::from_axis_angle(axis, rotation_speed * dt);
        self.config.camera_quaternion = self.config.camera_quaternion.multiply(&rotation);
    }
    
    /// Render fractal to image buffer
    pub fn render(&self, buffer: &mut RgbImage) {
        if self.config.fractal_type == FractalType::None {
            return;
        }
        
        let width = buffer.width() as usize;
        let height = buffer.height() as usize;
        let aspect = width as f32 / height as f32;
        let fov_y = 0.5_f32.tan();
        let fov_x = aspect * fov_y;
        
        // Parallel rendering
        let pixels: Vec<(u32, u32, Rgb<u8>)> = (0..height)
            .into_par_iter()
            .flat_map(|y| {
                let mut row_pixels = Vec::with_capacity(width);
                for x in 0..width {
                    let u = (2.0 * x as f32 / width as f32 - 1.0) * fov_x;
                    let v = (1.0 - 2.0 * y as f32 / height as f32) * fov_y;
                    
                    let direction = self.config.camera_quaternion
                        .rotate_vector(Vec3::new(u, v, -1.0).normalize());
                    let position = self.config.camera_quaternion
                        .rotate_vector(Vec3::new(0.0, 0.0, self.config.orbit_distance));
                    
                    let color = self.cast_ray(position, direction);
                    row_pixels.push((x as u32, y as u32, color));
                }
                row_pixels
            })
            .collect();
        
        for (x, y, color) in pixels {
            buffer.put_pixel(x, y, color);
        }
    }
    
    /// Cast a single ray and return color
    fn cast_ray(&self, origin: Vec3, direction: Vec3) -> Rgb<u8> {
        let mut position = origin;
        let mut travel = 0.0_f32;
        let mut orbit_trap = OrbitTrap::new();
        
        for step in 0..self.config.max_ray_steps {
            let (dist, trap) = self.distance_estimator(position);
            orbit_trap = trap;
            
            if dist <= self.config.epsilon {
                // Hit - color based on orbit trap
                let iter_ratio = step as f32 / self.config.max_ray_steps as f32;
                let dist_ratio = travel / self.config.max_distance;
                return self.shade_hit(orbit_trap, dist_ratio, iter_ratio);
            }
            
            position = position.add(&direction.scale(dist * 0.99));
            travel += dist;
            
            if travel >= self.config.max_distance {
                break;
            }
        }
        
        // Miss - return background color
        self.shade_background(direction)
    }
    
    /// Distance estimator for selected fractal type
    fn distance_estimator(&self, point: Vec3) -> (f32, OrbitTrap) {
        match self.config.fractal_type {
            FractalType::None => (1000.0, OrbitTrap::new()),
            FractalType::Mandelbox => self.de_mandelbox(point),
            FractalType::Mandelbulb => self.de_mandelbulb(point),
            FractalType::Klein => self.de_klein(point),
            FractalType::MengerSponge => self.de_menger(point),
            FractalType::Sierpinski => self.de_sierpinski(point),
            FractalType::QuaternionJulia => self.de_quaternion_julia(point),
        }
    }
    
    /// Mandelbox distance estimator
    fn de_mandelbox(&self, point: Vec3) -> (f32, OrbitTrap) {
        let max_iter = 5;
        let re_scale = 4.8;
        let mut s = point.scale(re_scale);
        let t = s;
        let scale = 0.25 * (0.075 * self.config.time).cos() - 2.1;
        let mut de_factor = 1.0_f32;
        let mut orbit_trap = OrbitTrap::new();
        let bvr = 12.0_f32.sqrt();
        
        for _ in 0..max_iter {
            // Box fold
            if s.x > 1.0 { s.x = 2.0 - s.x; } else if s.x < -1.0 { s.x = -2.0 - s.x; }
            if s.y > 1.0 { s.y = 2.0 - s.y; } else if s.y < -1.0 { s.y = -2.0 - s.y; }
            if s.z > 1.0 { s.z = 2.0 - s.z; } else if s.z < -1.0 { s.z = -2.0 - s.z; }
            
            // Sphere fold
            let r2 = s.length_sq();
            if r2 < 0.25 {
                s = s.scale(4.0);
                de_factor *= 4.0;
            } else if r2 < 1.0 {
                s = s.scale(1.0 / r2);
                de_factor /= r2;
            }
            
            // Orbit trap
            orbit_trap.update_min(s.scale(1.0 / bvr).sub(&self.config.reactive_bass).length() / 1.25, 0);
            orbit_trap.update_min(s.scale(1.0 / bvr).sub(&self.config.reactive_mids).length() / 1.25, 1);
            orbit_trap.update_min(s.scale(1.0 / bvr).sub(&self.config.reactive_high).length() / 1.25, 2);
            
            s = s.scale(scale).add(&t);
            de_factor = de_factor * scale.abs() + 1.0;
            
            if r2 > 12.0 { break; }
        }
        
        let dist = (s.length() - bvr) / de_factor.abs() / re_scale;
        (dist, orbit_trap)
    }
    
    /// Mandelbulb distance estimator
    fn de_mandelbulb(&self, point: Vec3) -> (f32, OrbitTrap) {
        let max_iter = 3;
        let re_scale = 1.85;
        let mut s = point.scale(re_scale);
        let t = s;
        let power = 9.0 + 2.0 * self.bound_reflect(0.0375 * self.config.time + 1.0, 1.0);
        let mut dr = 1.0_f32;
        let mut r = 0.0_f32;
        let mut orbit_trap = OrbitTrap::new();
        
        for _ in 0..max_iter {
            r = s.length();
            if r > 1.5 { break; }
            
            let theta = (s.z / r).acos();
            let phi = s.y.atan2(s.x);
            dr = r.powf(power - 1.0) * power * dr + 1.0;
            
            let new_r = r.powf(power);
            let new_theta = theta * power;
            let new_phi = phi * power;
            
            s = Vec3::new(
                new_r * new_theta.sin() * new_phi.cos(),
                new_r * new_theta.sin() * new_phi.sin(),
                new_r * new_theta.cos(),
            );
            s = s.add(&t);
            
            let reactive_center = self.config.reactive_high.add(&self.config.reactive_bass).scale(0.5);
            let orbit_dist = s.sub(&reactive_center).abs();
            orbit_trap.r = orbit_trap.r.min(orbit_dist.x / 1.25);
            orbit_trap.g = orbit_trap.g.min(orbit_dist.y / 1.25);
            orbit_trap.b = orbit_trap.b.min(orbit_dist.z / 1.25);
        }
        
        let dist = (0.5 * r.ln() * r / dr).min(3.5) / re_scale;
        (dist, orbit_trap)
    }
    
    /// Klein bottle inspired IFS
    fn de_klein(&self, point: Vec3) -> (f32, OrbitTrap) {
        let max_iter = 3;
        let re_scale = 0.8;
        let mut s = point.scale(re_scale);
        let t = s;
        let anim = 1.275 + 0.085 * (0.2 * self.config.time).sin();
        let mut scale = 1.0_f32;
        let theta = 0.1 * self.config.time;
        let ct = theta.cos();
        let st = theta.sin();
        let mut orbit_trap = OrbitTrap::new();
        
        for i in 0..max_iter {
            if i == 2 {
                let new_x = s.x * ct - s.y * st;
                let new_y = s.x * st + s.y * ct;
                s.x = new_x;
                s.y = new_y;
            }
            
            // Mod fold
            s.x = -1.0 + 2.0 * ((0.5 * s.x + 0.5) - (0.5 * s.x + 0.5).floor());
            s.y = -1.0 + 2.0 * ((0.5 * s.y + 0.5) - (0.5 * s.y + 0.5).floor());
            s.z = -1.0 + 2.0 * ((0.5 * s.z + 0.5) - (0.5 * s.z + 0.5).floor());
            
            let r2 = s.length_sq();
            let k = anim / r2;
            s = s.scale(k);
            scale *= k;
            
            let reactive_center = self.config.reactive_high.add(&self.config.reactive_bass).scale(0.5);
            let orbit_dist = s.sub(&reactive_center).abs();
            orbit_trap.r = orbit_trap.r.min(orbit_dist.x);
            orbit_trap.g = orbit_trap.g.min(orbit_dist.y);
            orbit_trap.b = orbit_trap.b.min(orbit_dist.z);
        }
        
        let dist = ((0.25 * s.z.abs() / scale) / re_scale).max((t.scale(1.0 / re_scale).length() - 0.62));
        (dist, orbit_trap)
    }
    
    /// Menger Sponge distance estimator
    fn de_menger(&self, point: Vec3) -> (f32, OrbitTrap) {
        let max_iter = 4;
        let re_scale = 1.32;
        let mut s = point.scale(re_scale).add(&Vec3::new(0.5, 0.5, 0.5));
        let mut orbit_trap = OrbitTrap::new();
        
        let xx = (s.x - 0.5).abs() - 0.5;
        let yy = (s.y - 0.5).abs() - 0.5;
        let zz = (s.z - 0.5).abs() - 0.5;
        let mut d = xx.max(yy.max(zz));
        let mut p = 1.0_f32;
        
        orbit_trap.r = (xx / 1.2).abs();
        orbit_trap.g = (yy / 1.2).abs();
        orbit_trap.b = (zz / 1.2).abs();
        
        for _ in 0..max_iter {
            p *= 3.0;
            let xa = (s.x * p) % 3.0;
            let ya = (s.y * p) % 3.0;
            let za = (s.z * p) % 3.0;
            
            let xx = 0.5 - (xa - 1.5).abs();
            let yy = 0.5 - (ya - 1.5).abs();
            let zz = 0.5 - (za - 1.5).abs();
            
            let d1 = xx.max(zz).min(xx.max(yy).min(yy.max(zz))) / p;
            d = d.max(d1);
            
            let q = Vec3::new(xx, yy, zz);
            orbit_trap.r = orbit_trap.r.max(q.dot(&self.config.reactive_bass).abs());
            orbit_trap.g = orbit_trap.g.max(q.dot(&self.config.reactive_mids).abs());
            orbit_trap.b = orbit_trap.b.max(q.dot(&self.config.reactive_high).abs());
        }
        
        (d / re_scale, orbit_trap)
    }
    
    /// Sierpinski Tetrahedron distance estimator
    fn de_sierpinski(&self, point: Vec3) -> (f32, OrbitTrap) {
        let max_iter = 8;
        let scale = 2.0_f32;
        let re_scale = 1.375;
        let mut s = point.scale(re_scale);
        let center = Vec3::new(0.5_f32.sqrt(), 0.3_f32.sqrt(), 0.2_f32.sqrt());
        let mut de_factor = 1.0_f32;
        let mut orbit_trap = OrbitTrap::new();
        
        for _ in 0..max_iter {
            let r2 = s.length_sq();
            if r2 > 1000.0 { break; }
            
            // Fold
            if s.x + s.y < 0.0 { let x1 = -s.y; s.y = -s.x; s.x = x1; }
            if s.x + s.z < 0.0 { let x1 = -s.z; s.z = -s.x; s.x = x1; }
            if s.y + s.z < 0.0 { let y1 = -s.z; s.z = -s.y; s.y = y1; }
            
            s = s.scale(scale).sub(&center.scale(scale - 1.0));
            
            orbit_trap.update_min(s.sub(&self.config.reactive_bass).length() / 2.0, 0);
            orbit_trap.update_min(s.sub(&self.config.reactive_mids).length() / 2.0, 1);
            orbit_trap.update_min(s.sub(&self.config.reactive_high).length() / 2.0, 2);
            
            de_factor *= scale;
        }
        
        ((s.length() - 2.0) / de_factor / re_scale, orbit_trap)
    }
    
    /// Quaternion Julia distance estimator
    fn de_quaternion_julia(&self, point: Vec3) -> (f32, OrbitTrap) {
        let max_iter = 6;
        let re_scale = 1.85;
        let power = 4.0 + (0.025 * self.config.time).sin();
        let mut dr = 1.0_f32;
        let mut orbit_trap = OrbitTrap::new();
        
        // Quaternion from position
        let mut qr = 0.0_f32;
        let mut qi = point.x * re_scale;
        let mut qj = point.y * re_scale;
        let mut qk = point.z * re_scale;
        
        // Julia constant from reactive vectors
        let cr = 0.5 * (self.config.time * 0.1).sin();
        let ci = self.config.reactive_bass.x * 0.3;
        let cj = self.config.reactive_mids.y * 0.3;
        let ck = self.config.reactive_high.z * 0.3;
        
        for _ in 0..max_iter {
            let r = (qr * qr + qi * qi + qj * qj + qk * qk).sqrt();
            if r > 1.5 { break; }
            
            dr = power * r.powf(power - 1.0) * dr;
            
            // Power of quaternion using spherical coords
            let phi = (qr / r).acos();
            let new_r = r.powf(power);
            let new_phi = phi * power;
            
            let sin_phi = new_phi.sin();
            let len_v = (qi * qi + qj * qj + qk * qk).sqrt();
            let factor = if len_v > 0.0001 { new_r * sin_phi / len_v } else { 0.0 };
            
            qr = new_r * new_phi.cos() + cr;
            qi = qi * factor + ci;
            qj = qj * factor + cj;
            qk = qk * factor + ck;
            
            let reactive_center = self.config.reactive_high.add(&self.config.reactive_bass).scale(0.5);
            let orbit_dist = Vec3::new(qi, qj, qk).sub(&reactive_center).abs();
            orbit_trap.r = orbit_trap.r.min(orbit_dist.x.sqrt().sqrt() / 3.5);
            orbit_trap.g = orbit_trap.g.min(orbit_dist.y.sqrt().sqrt() / 3.5);
            orbit_trap.b = orbit_trap.b.min(orbit_dist.z.sqrt().sqrt() / 3.5);
        }
        
        let r = (qr * qr + qi * qi + qj * qj + qk * qk).sqrt();
        let dist = 0.6 * (r.ln() * r / dr).min(3.5) / re_scale;
        (dist, orbit_trap)
    }
    
    /// Helper for bounded reflection
    fn bound_reflect(&self, x: f32, b: f32) -> f32 {
        let r = (x + b) % (4.0 * b);
        if r < 2.0 * b {
            r - b
        } else {
            3.0 * b - r
        }
    }
    
    /// Shade a hit point based on orbit trap
    fn shade_hit(&self, trap: OrbitTrap, dist_ratio: f32, iter_ratio: f32) -> Rgb<u8> {
        let brightness = (1.0 - dist_ratio).powf(1.2) * (1.0 - iter_ratio).powf(2.5);
        
        let r = ((1.0 - trap.r) * brightness * 255.0).clamp(0.0, 255.0) as u8;
        let g = ((1.0 - trap.g) * brightness * 255.0).clamp(0.0, 255.0) as u8;
        let b = ((1.0 - trap.b) * brightness * 255.0).clamp(0.0, 255.0) as u8;
        
        Rgb([r, g, b])
    }
    
    /// Shade background
    fn shade_background(&self, direction: Vec3) -> Rgb<u8> {
        // Audio-reactive starfield background
        let sin_dir = Vec3::new(
            (100.0 * direction.x).sin(),
            (100.0 * direction.y).sin(),
            (100.0 * direction.z).sin(),
        );
        
        let r = (-2.9 * (self.config.reactive_bass.scale(PI).add(&Vec3::new(1.0, 1.0, 1.0)).sub(&sin_dir)).length()).exp();
        let g = (-2.9 * (self.config.reactive_mids.scale(2.718281828).add(&Vec3::new(1.3, 1.3, 1.3)).sub(&sin_dir)).length()).exp();
        let b = (-2.9 * (self.config.reactive_high.scale(9.6).add(&Vec3::new(117.69, 117.69, 117.69)).sub(&sin_dir)).length()).exp();
        
        let scale = 0.54;
        Rgb([
            (r * scale * 255.0).clamp(0.0, 255.0) as u8,
            (g * scale * 255.0).clamp(0.0, 255.0) as u8,
            (b * scale * 255.0).clamp(0.0, 255.0) as u8,
        ])
    }
}
