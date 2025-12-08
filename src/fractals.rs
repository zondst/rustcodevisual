//! Fractal Renderer for Particle Studio RS
//! CPU-based raymarching implementation inspired by fractal_sugar
//! Supports 7 fractal types with audio-reactive coloring

use image::Rgb;
use std::f32::consts::PI;

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    pub fn scale(&self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    pub fn length_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct OrbitTrap {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl OrbitTrap {
    pub fn new() -> Self {
        Self {
            r: 100.0,
            g: 100.0,
            b: 100.0,
        }
    }
    pub fn update_min(&mut self, val: f32, channel: usize) {
        match channel {
            0 => self.r = self.r.min(val),
            1 => self.g = self.g.min(val),
            2 => self.b = self.b.min(val),
            _ => {}
        }
    }
}

#[allow(dead_code)]
pub struct FractalConfig {
    pub reactive_bass: Vec3,
    pub reactive_mids: Vec3,
    pub reactive_high: Vec3,
    pub time: f32,
}

#[allow(dead_code)]
pub struct FractalRenderer {
    pub config: FractalConfig,
}

impl FractalRenderer {
    #[allow(dead_code)]
    fn de_menger(&self, point: Vec3) -> (f32, OrbitTrap) {
        let max_iter = 8;
        let re_scale = 1.32;
        let s = point.scale(re_scale).add(&Vec3::new(0.5, 0.5, 0.5));
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
            if r2 > 1000.0 {
                break;
            }

            // Fold
            if s.x + s.y < 0.0 {
                let x1 = -s.y;
                s.y = -s.x;
                s.x = x1;
            }
            if s.x + s.z < 0.0 {
                let x1 = -s.z;
                s.z = -s.x;
                s.x = x1;
            }
            if s.y + s.z < 0.0 {
                let y1 = -s.z;
                s.z = -s.y;
                s.y = y1;
            }

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
            if r > 1.5 {
                break;
            }

            dr = power * r.powf(power - 1.0) * dr;

            // Power of quaternion using spherical coords
            let phi = (qr / r).acos();
            let new_r = r.powf(power);
            let new_phi = phi * power;

            let sin_phi = new_phi.sin();
            let len_v = (qi * qi + qj * qj + qk * qk).sqrt();
            let factor = if len_v > 0.0001 {
                new_r * sin_phi / len_v
            } else {
                0.0
            };

            qr = new_r * new_phi.cos() + cr;
            qi = qi * factor + ci;
            qj = qj * factor + cj;
            qk = qk * factor + ck;

            let reactive_center = self
                .config
                .reactive_high
                .add(&self.config.reactive_bass)
                .scale(0.5);
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

        let r = (-2.9
            * (self
                .config
                .reactive_bass
                .scale(PI)
                .add(&Vec3::new(1.0, 1.0, 1.0))
                .sub(&sin_dir))
            .length())
        .exp();
        let g = (-2.9
            * (self
                .config
                .reactive_mids
                .scale(2.718281828)
                .add(&Vec3::new(1.3, 1.3, 1.3))
                .sub(&sin_dir))
            .length())
        .exp();
        let b = (-2.9
            * (self
                .config
                .reactive_high
                .scale(9.6)
                .add(&Vec3::new(117.69, 117.69, 117.69))
                .sub(&sin_dir))
            .length())
        .exp();

        let scale = 0.54;
        Rgb([
            (r * scale * 255.0).clamp(0.0, 255.0) as u8,
            (g * scale * 255.0).clamp(0.0, 255.0) as u8,
            (b * scale * 255.0).clamp(0.0, 255.0) as u8,
        ])
    }
}
