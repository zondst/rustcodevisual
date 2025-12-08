//! Post-Processing Effects for Particle Studio RS
//! CPU-based image effects: bloom, blur, vignette, chromatic aberration, etc.

use image::Rgb;
use std::f32::consts::PI;

use crate::audio::AudioState;
use crate::config::VisualConfig;

/// Frame buffer for rendering
use image::ImageBuffer;
pub type FrameBuffer = ImageBuffer<Rgb<u8>, Vec<u8>>;

/// Post-processor with all effects
#[allow(dead_code)]
pub struct PostProcessor {
    width: u32,
    height: u32,

    // Cached masks
    vignette_mask: Vec<f32>,

    // Motion blur buffer
    prev_frame: Option<Vec<[f32; 3]>>,

    // Echo effect buffer
    echo_buffer: Option<Vec<[f32; 3]>>,

    // Scanline mask
    scanline_mask: Vec<f32>,

    // Color shift phase
    color_shift_phase: f32,
}

impl PostProcessor {
    pub fn new(width: u32, height: u32) -> Self {
        let vignette_mask = Self::create_vignette_mask(width, height);
        let scanline_mask = Self::create_scanline_mask(height);

        Self {
            width,
            height,
            vignette_mask,
            prev_frame: None,
            echo_buffer: None,
            scanline_mask,
            color_shift_phase: 0.0,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if self.width != width || self.height != height {
            self.width = width;
            self.height = height;
            self.vignette_mask = Self::create_vignette_mask(width, height);
            self.scanline_mask = Self::create_scanline_mask(height);
            self.prev_frame = None;
            self.echo_buffer = None;
        }
    }

    fn create_vignette_mask(width: u32, height: u32) -> Vec<f32> {
        let mut mask = vec![1.0f32; (width * height) as usize];
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt();

        for y in 0..height {
            for x in 0..width {
                let dx = (x as f32 - cx) / cx;
                let dy = (y as f32 - cy) / cy;
                let dist = (dx * dx + dy * dy).sqrt();

                // Smooth vignette falloff
                let v = 1.0 - ((dist - 0.5) * 1.2).max(0.0).min(1.0);
                let v = v.powf(1.5);

                mask[(y * width + x) as usize] = v;
            }
        }

        mask
    }

    fn create_scanline_mask(height: u32) -> Vec<f32> {
        (0..height)
            .map(|y| if y % 2 == 0 { 0.85 } else { 1.0 })
            .collect()
    }

    /// Process a frame with all enabled effects
    pub fn process(&mut self, buffer: &mut FrameBuffer, config: &VisualConfig, audio: &AudioState) {
        let width = buffer.width();
        let height = buffer.height();

        // Convert to float buffer for processing
        let mut float_buffer: Vec<[f32; 3]> = buffer
            .pixels()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();

        // Apply MilkDrop effects first
        if config.echo_enabled {
            self.apply_echo(&mut float_buffer, width, height, config);
        }

        if config.kaleidoscope_enabled {
            self.apply_kaleidoscope(
                &mut float_buffer,
                width,
                height,
                config.kaleidoscope_segments,
            );
        }

        if config.radial_blur_enabled && config.radial_blur_amount > 0.01 {
            let amount = config.radial_blur_amount * (1.0 + audio.beat * 0.5);
            self.apply_radial_blur(&mut float_buffer, width, height, amount);
        }

        // Chromatic aberration
        if config.chromatic_aberration > 0.0 {
            self.apply_chromatic_aberration(
                &mut float_buffer,
                width,
                height,
                config.chromatic_aberration,
            );
        }

        // Scanlines
        if config.scanlines {
            self.apply_scanlines(&mut float_buffer, width, height, config.scanline_intensity);
        }

        // Color shift
        if config.color_shift_enabled {
            self.apply_color_shift(&mut float_buffer, audio.amplitude);
        }

        // Bloom
        if config.bloom_enabled && config.bloom_intensity > 0.0 {
            let intensity = config.bloom_intensity * (1.0 + audio.beat * 0.5);
            self.apply_bloom(
                &mut float_buffer,
                width,
                height,
                intensity,
                config.bloom_threshold,
                config.bloom_radius,
            );
        }

        // Motion blur
        if config.motion_blur > 0.0 {
            self.apply_motion_blur(&mut float_buffer, config.motion_blur);
        }

        // Vignette
        if config.vignette_strength > 0.0 {
            self.apply_vignette(&mut float_buffer, width, height, config.vignette_strength);
        }

        // Film grain
        if config.film_grain > 0.0 {
            self.apply_film_grain(&mut float_buffer, config.film_grain);
        }

        // Tone mapping
        self.apply_tone_mapping(&mut float_buffer, config);

        // Convert back to u8 buffer
        for (i, pixel) in buffer.pixels_mut().enumerate() {
            let [r, g, b] = float_buffer[i];
            pixel[0] = r.clamp(0.0, 255.0) as u8;
            pixel[1] = g.clamp(0.0, 255.0) as u8;
            pixel[2] = b.clamp(0.0, 255.0) as u8;
        }
    }

    // ========================================================================
    // Individual Effects
    // ========================================================================

    fn apply_bloom(
        &self,
        buffer: &mut Vec<[f32; 3]>,
        width: u32,
        height: u32,
        intensity: f32,
        threshold: f32,
        radius: f32,
    ) {
        // Downscale factor for performance
        let scale = 4u32;
        let small_w = width / scale;
        let small_h = height / scale;

        // Extract bright areas and downscale
        let mut bright: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; (small_w * small_h) as usize];

        for sy in 0..small_h {
            for sx in 0..small_w {
                let x = sx * scale;
                let y = sy * scale;
                let idx = (y * width + x) as usize;

                if idx < buffer.len() {
                    let [r, g, b] = buffer[idx];
                    let brightness = (r + g + b) / (3.0 * 255.0);

                    if brightness > threshold {
                        let factor = (brightness - threshold) / (1.0 - threshold + 0.001);
                        bright[(sy * small_w + sx) as usize] = [r * factor, g * factor, b * factor];
                    }
                }
            }
        }

        // Apply gaussian blur to bright areas
        let sigma = (radius / scale as f32).max(1.0);
        let kernel_size = ((sigma * 3.0) as usize * 2 + 1).min(15);
        let kernel = Self::gaussian_kernel(kernel_size, sigma);

        // Horizontal blur
        let mut temp = bright.clone();
        Self::blur_horizontal(&bright, &mut temp, small_w, small_h, &kernel);

        // Vertical blur
        Self::blur_vertical(&temp, &mut bright, small_w, small_h, &kernel);

        // Add bloom back to original (upscaled)
        for y in 0..height {
            for x in 0..width {
                let sx = (x / scale).min(small_w - 1);
                let sy = (y / scale).min(small_h - 1);
                let bloom_idx = (sy * small_w + sx) as usize;
                let idx = (y * width + x) as usize;

                if idx < buffer.len() && bloom_idx < bright.len() {
                    let [br, bg, bb] = bright[bloom_idx];
                    buffer[idx][0] += br * intensity;
                    buffer[idx][1] += bg * intensity;
                    buffer[idx][2] += bb * intensity;
                }
            }
        }
    }

    /// Multi-pass bloom with mipchain (inspired by WebGPU-Bloom)
    /// Provides higher quality bloom with soft threshold transition
    pub fn apply_bloom_multipass(
        &self,
        buffer: &mut Vec<[f32; 3]>,
        width: u32,
        height: u32,
        intensity: f32,
        threshold: f32,
        knee: f32,
        mip_levels: u32,
    ) {
        let mip_levels = mip_levels.clamp(2, 7) as usize;

        // Create mipchain buffers
        let mut mips: Vec<Vec<[f32; 3]>> = Vec::with_capacity(mip_levels);
        let mut mip_widths: Vec<u32> = Vec::with_capacity(mip_levels);
        let mut mip_heights: Vec<u32> = Vec::with_capacity(mip_levels);

        let mut mip_w = width / 2;
        let mut mip_h = height / 2;

        for _ in 0..mip_levels {
            mips.push(vec![[0.0, 0.0, 0.0]; (mip_w * mip_h) as usize]);
            mip_widths.push(mip_w);
            mip_heights.push(mip_h);
            mip_w = (mip_w / 2).max(1);
            mip_h = (mip_h / 2).max(1);
        }

        // Prefilter pass: Extract bright pixels with soft knee
        let knee_offset = threshold - knee;
        let knee_scale = if knee > 0.0 { 0.25 / knee } else { 0.0 };

        let first_mip_w = mip_widths[0];
        let first_mip_h = mip_heights[0];

        for sy in 0..first_mip_h {
            for sx in 0..first_mip_w {
                let x = sx * 2;
                let y = sy * 2;
                let idx = (y * width + x) as usize;

                if idx < buffer.len() {
                    let [r, g, b] = buffer[idx];
                    let brightness = (r + g + b) / (3.0 * 255.0);

                    // Soft threshold with knee
                    let soft = (brightness - knee_offset).max(0.0);
                    let soft = soft * soft * knee_scale;
                    let contribution = soft.max(brightness - threshold).max(0.0);
                    let factor = contribution / (brightness + 0.001);

                    mips[0][(sy * first_mip_w + sx) as usize] =
                        [r * factor, g * factor, b * factor];
                }
            }
        }

        // Downsample passes
        for level in 1..mip_levels {
            let src_w = mip_widths[level - 1];
            let src_h = mip_heights[level - 1];
            let dst_w = mip_widths[level];
            let dst_h = mip_heights[level];

            let src = mips[level - 1].clone();
            let dst = &mut mips[level];

            for dy in 0..dst_h {
                for dx in 0..dst_w {
                    let sx = (dx * 2).min(src_w - 1);
                    let sy = (dy * 2).min(src_h - 1);

                    // 2x2 box filter
                    let mut sum = [0.0f32; 3];
                    let mut count = 0;

                    for oy in 0..2u32 {
                        for ox in 0..2u32 {
                            let sample_x = (sx + ox).min(src_w - 1);
                            let sample_y = (sy + oy).min(src_h - 1);
                            let src_idx = (sample_y * src_w + sample_x) as usize;

                            if src_idx < src.len() {
                                sum[0] += src[src_idx][0];
                                sum[1] += src[src_idx][1];
                                sum[2] += src[src_idx][2];
                                count += 1;
                            }
                        }
                    }

                    if count > 0 {
                        let divisor = count as f32;
                        dst[(dy * dst_w + dx) as usize] =
                            [sum[0] / divisor, sum[1] / divisor, sum[2] / divisor];
                    }
                }
            }

            // Apply blur to this mip level
            let sigma = 1.5;
            let kernel = Self::gaussian_kernel(5, sigma);
            let mut temp = dst.clone();
            Self::blur_horizontal(dst, &mut temp, dst_w, dst_h, &kernel);
            Self::blur_vertical(&temp, dst, dst_w, dst_h, &kernel);
        }

        // Upsample and combine passes
        let combine_constant = 0.68_f32;

        for level in (0..mip_levels - 1).rev() {
            let dst_w = mip_widths[level];
            let dst_h = mip_heights[level];
            let src_w = mip_widths[level + 1];
            let src_h = mip_heights[level + 1];

            let src = mips[level + 1].clone();
            let dst = &mut mips[level];

            for dy in 0..dst_h {
                for dx in 0..dst_w {
                    // Bilinear sample from smaller mip
                    let sx = (dx as f32 / 2.0).min((src_w - 1) as f32);
                    let sy = (dy as f32 / 2.0).min((src_h - 1) as f32);

                    let x0 = sx as u32;
                    let y0 = sy as u32;
                    let x1 = (x0 + 1).min(src_w - 1);
                    let y1 = (y0 + 1).min(src_h - 1);

                    let fx = sx - x0 as f32;
                    let fy = sy - y0 as f32;

                    let idx00 = (y0 * src_w + x0) as usize;
                    let idx10 = (y0 * src_w + x1) as usize;
                    let idx01 = (y1 * src_w + x0) as usize;
                    let idx11 = (y1 * src_w + x1) as usize;

                    let mut upsampled = [0.0f32; 3];
                    for c in 0..3 {
                        let v00 = src.get(idx00).map(|p| p[c]).unwrap_or(0.0);
                        let v10 = src.get(idx10).map(|p| p[c]).unwrap_or(0.0);
                        let v01 = src.get(idx01).map(|p| p[c]).unwrap_or(0.0);
                        let v11 = src.get(idx11).map(|p| p[c]).unwrap_or(0.0);

                        let top = v00 * (1.0 - fx) + v10 * fx;
                        let bottom = v01 * (1.0 - fx) + v11 * fx;
                        upsampled[c] = top * (1.0 - fy) + bottom * fy;
                    }

                    // Combine with current level
                    let dst_idx = (dy * dst_w + dx) as usize;
                    if dst_idx < dst.len() {
                        dst[dst_idx][0] = dst[dst_idx][0] * (1.0 - combine_constant)
                            + upsampled[0] * combine_constant;
                        dst[dst_idx][1] = dst[dst_idx][1] * (1.0 - combine_constant)
                            + upsampled[1] * combine_constant;
                        dst[dst_idx][2] = dst[dst_idx][2] * (1.0 - combine_constant)
                            + upsampled[2] * combine_constant;
                    }
                }
            }
        }

        // Add final bloom to original buffer
        let bloom_src = &mips[0];
        let bloom_w = mip_widths[0];
        let bloom_h = mip_heights[0];

        for y in 0..height {
            for x in 0..width {
                let bx = (x / 2).min(bloom_w - 1);
                let by = (y / 2).min(bloom_h - 1);
                let bloom_idx = (by * bloom_w + bx) as usize;
                let idx = (y * width + x) as usize;

                if idx < buffer.len() && bloom_idx < bloom_src.len() {
                    buffer[idx][0] += bloom_src[bloom_idx][0] * intensity;
                    buffer[idx][1] += bloom_src[bloom_idx][1] * intensity;
                    buffer[idx][2] += bloom_src[bloom_idx][2] * intensity;
                }
            }
        }
    }

    fn gaussian_kernel(size: usize, sigma: f32) -> Vec<f32> {
        let mut kernel = vec![0.0f32; size];
        let center = size as f32 / 2.0;
        let mut sum = 0.0;

        for i in 0..size {
            let x = i as f32 - center;
            kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
            sum += kernel[i];
        }

        // Normalize
        for k in &mut kernel {
            *k /= sum;
        }

        kernel
    }

    fn blur_horizontal(
        src: &[[f32; 3]],
        dst: &mut [[f32; 3]],
        width: u32,
        height: u32,
        kernel: &[f32],
    ) {
        let half = kernel.len() / 2;

        for y in 0..height {
            for x in 0..width {
                let mut r = 0.0f32;
                let mut g = 0.0f32;
                let mut b = 0.0f32;

                for (ki, &kv) in kernel.iter().enumerate() {
                    let sx = (x as i32 + ki as i32 - half as i32)
                        .max(0)
                        .min(width as i32 - 1) as u32;
                    let idx = (y * width + sx) as usize;

                    r += src[idx][0] * kv;
                    g += src[idx][1] * kv;
                    b += src[idx][2] * kv;
                }

                dst[(y * width + x) as usize] = [r, g, b];
            }
        }
    }

    fn blur_vertical(
        src: &[[f32; 3]],
        dst: &mut [[f32; 3]],
        width: u32,
        height: u32,
        kernel: &[f32],
    ) {
        let half = kernel.len() / 2;

        for y in 0..height {
            for x in 0..width {
                let mut r = 0.0f32;
                let mut g = 0.0f32;
                let mut b = 0.0f32;

                for (ki, &kv) in kernel.iter().enumerate() {
                    let sy = (y as i32 + ki as i32 - half as i32)
                        .max(0)
                        .min(height as i32 - 1) as u32;
                    let idx = (sy * width + x) as usize;

                    r += src[idx][0] * kv;
                    g += src[idx][1] * kv;
                    b += src[idx][2] * kv;
                }

                dst[(y * width + x) as usize] = [r, g, b];
            }
        }
    }

    fn apply_motion_blur(&mut self, buffer: &mut Vec<[f32; 3]>, amount: f32) {
        let amount = amount.clamp(0.0, 0.9);

        if let Some(ref prev) = self.prev_frame {
            if prev.len() == buffer.len() {
                for (i, pixel) in buffer.iter_mut().enumerate() {
                    pixel[0] = pixel[0] * (1.0 - amount) + prev[i][0] * amount;
                    pixel[1] = pixel[1] * (1.0 - amount) + prev[i][1] * amount;
                    pixel[2] = pixel[2] * (1.0 - amount) + prev[i][2] * amount;
                }
            }
        }

        self.prev_frame = Some(buffer.clone());
    }

    fn apply_vignette(&self, buffer: &mut Vec<[f32; 3]>, width: u32, height: u32, strength: f32) {
        for (i, pixel) in buffer.iter_mut().enumerate() {
            let mask_val = self.vignette_mask.get(i).copied().unwrap_or(1.0);
            let factor = 1.0 - (1.0 - mask_val) * strength;

            pixel[0] *= factor;
            pixel[1] *= factor;
            pixel[2] *= factor;
        }
    }

    fn apply_film_grain(&self, buffer: &mut Vec<[f32; 3]>, amount: f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let noise_scale = amount * 30.0;

        for pixel in buffer.iter_mut() {
            let noise = rng.gen_range(-noise_scale..noise_scale);
            pixel[0] += noise;
            pixel[1] += noise;
            pixel[2] += noise;
        }
    }

    fn apply_chromatic_aberration(
        &self,
        buffer: &mut Vec<[f32; 3]>,
        width: u32,
        height: u32,
        amount: f32,
    ) {
        let offset = ((amount * width.min(height) as f32 * 0.01).max(1.0)) as i32;
        let original = buffer.clone();

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;

                // Red channel - shift left
                let rx = (x as i32 - offset).max(0).min(width as i32 - 1) as u32;
                let r_idx = (y * width + rx) as usize;
                buffer[idx][0] = original[r_idx][0];

                // Green - no shift
                // (already correct)

                // Blue channel - shift right
                let bx = (x as i32 + offset).max(0).min(width as i32 - 1) as u32;
                let b_idx = (y * width + bx) as usize;
                buffer[idx][2] = original[b_idx][2];
            }
        }
    }

    fn apply_scanlines(&self, buffer: &mut Vec<[f32; 3]>, width: u32, height: u32, intensity: f32) {
        for y in 0..height {
            let scanline_factor = 1.0 - intensity * (1.0 - self.scanline_mask[y as usize]);

            for x in 0..width {
                let idx = (y * width + x) as usize;
                buffer[idx][0] *= scanline_factor;
                buffer[idx][1] *= scanline_factor;
                buffer[idx][2] *= scanline_factor;
            }
        }
    }

    fn apply_color_shift(&mut self, buffer: &mut Vec<[f32; 3]>, amplitude: f32) {
        self.color_shift_phase += 0.1 * amplitude;
        let shift = self.color_shift_phase.sin() * 0.3;
        let abs_shift = shift.abs();

        for pixel in buffer.iter_mut() {
            let r = pixel[0];
            let b = pixel[2];

            pixel[0] = r * (1.0 - abs_shift) + b * abs_shift;
            pixel[2] = b * (1.0 - abs_shift) + r * abs_shift;
        }
    }

    fn apply_echo(
        &mut self,
        buffer: &mut Vec<[f32; 3]>,
        width: u32,
        height: u32,
        config: &VisualConfig,
    ) {
        let zoom = config.echo_zoom;
        let rotation = config.echo_rotation;
        let alpha = config.echo_alpha;

        if self.echo_buffer.is_none() {
            self.echo_buffer = Some(buffer.clone());
            return;
        }

        let echo = self.echo_buffer.as_ref().unwrap();
        if echo.len() != buffer.len() {
            self.echo_buffer = Some(buffer.clone());
            return;
        }

        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;

        let cos_r = rotation.cos();
        let sin_r = rotation.sin();

        let mut result = buffer.clone();

        for y in 0..height {
            for x in 0..width {
                // Transform coordinates (zoom out and rotate)
                let dx = (x as f32 - cx) / zoom;
                let dy = (y as f32 - cy) / zoom;

                let rx = dx * cos_r - dy * sin_r + cx;
                let ry = dx * sin_r + dy * cos_r + cy;

                if rx >= 0.0 && rx < width as f32 && ry >= 0.0 && ry < height as f32 {
                    let src_idx = (ry as u32 * width + rx as u32) as usize;
                    let dst_idx = (y * width + x) as usize;

                    if src_idx < echo.len() {
                        result[dst_idx][0] =
                            buffer[dst_idx][0] * (1.0 - alpha) + echo[src_idx][0] * alpha;
                        result[dst_idx][1] =
                            buffer[dst_idx][1] * (1.0 - alpha) + echo[src_idx][1] * alpha;
                        result[dst_idx][2] =
                            buffer[dst_idx][2] * (1.0 - alpha) + echo[src_idx][2] * alpha;
                    }
                }
            }
        }

        *buffer = result.clone();
        self.echo_buffer = Some(result);
    }

    fn apply_kaleidoscope(
        &self,
        buffer: &mut Vec<[f32; 3]>,
        width: u32,
        height: u32,
        segments: usize,
    ) {
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let segment_angle = 2.0 * PI / segments as f32;

        let original = buffer.clone();

        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;

                let r = (dx * dx + dy * dy).sqrt();
                let mut theta = dy.atan2(dx) + PI;

                // Map to first segment
                theta = (theta % segment_angle - segment_angle / 2.0).abs();

                let new_x = (r * theta.cos() + cx).max(0.0).min(width as f32 - 1.0);
                let new_y = (r * theta.sin() + cy).max(0.0).min(height as f32 - 1.0);

                let src_idx = (new_y as u32 * width + new_x as u32) as usize;
                let dst_idx = (y * width + x) as usize;

                if src_idx < original.len() {
                    buffer[dst_idx] = original[src_idx];
                }
            }
        }
    }

    fn apply_radial_blur(&self, buffer: &mut Vec<[f32; 3]>, width: u32, height: u32, amount: f32) {
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let samples = 4;

        let original = buffer.clone();

        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt() / (cx.max(cy));

                let blur_amount = amount * dist;

                if blur_amount < 0.01 {
                    continue;
                }

                let mut r = 0.0f32;
                let mut g = 0.0f32;
                let mut b = 0.0f32;

                for s in 0..samples {
                    let scale = 1.0 + blur_amount * (s as f32 / samples as f32 - 0.5);
                    let sx = (dx * scale + cx).max(0.0).min(width as f32 - 1.0) as u32;
                    let sy = (dy * scale + cy).max(0.0).min(height as f32 - 1.0) as u32;

                    let idx = (sy * width + sx) as usize;
                    if idx < original.len() {
                        r += original[idx][0];
                        g += original[idx][1];
                        b += original[idx][2];
                    }
                }

                let idx = (y * width + x) as usize;
                buffer[idx] = [r / samples as f32, g / samples as f32, b / samples as f32];
            }
        }
    }

    fn apply_tone_mapping(&self, buffer: &mut Vec<[f32; 3]>, config: &VisualConfig) {
        let exposure = config.exposure;
        let contrast = config.contrast;
        let saturation = config.saturation;
        let gamma = config.gamma;

        for pixel in buffer.iter_mut() {
            // Exposure
            pixel[0] *= exposure;
            pixel[1] *= exposure;
            pixel[2] *= exposure;

            // Contrast
            let mid = 128.0;
            pixel[0] = (pixel[0] - mid) * contrast + mid;
            pixel[1] = (pixel[1] - mid) * contrast + mid;
            pixel[2] = (pixel[2] - mid) * contrast + mid;

            // Saturation
            if saturation != 1.0 {
                let gray = (pixel[0] + pixel[1] + pixel[2]) / 3.0;
                pixel[0] = gray + (pixel[0] - gray) * saturation;
                pixel[1] = gray + (pixel[1] - gray) * saturation;
                pixel[2] = gray + (pixel[2] - gray) * saturation;
            }

            // Gamma
            if gamma != 1.0 {
                pixel[0] = 255.0 * (pixel[0].max(0.0) / 255.0).powf(gamma);
                pixel[1] = 255.0 * (pixel[1].max(0.0) / 255.0).powf(gamma);
                pixel[2] = 255.0 * (pixel[2].max(0.0) / 255.0).powf(gamma);
            }
        }
    }
}

/// Background renderer with animated nebula effect
#[allow(dead_code)]
pub struct BackgroundRenderer {
    width: u32,
    height: u32,
    time: f32,
}

impl BackgroundRenderer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            time: 0.0,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    pub fn update(&mut self, dt: f32, amplitude: f32) {
        self.time += dt * 0.3 * (1.0 + amplitude * 0.5);
    }

    pub fn render(
        &self,
        buffer: &mut FrameBuffer,
        colors: &crate::config::ColorScheme,
        animated: bool,
        intensity: f32,
        audio: &AudioState,
    ) {
        let bg = colors.background;

        if !animated {
            for pixel in buffer.pixels_mut() {
                pixel[0] = bg[0];
                pixel[1] = bg[1];
                pixel[2] = bg[2];
            }
            return;
        }

        let dark = colors.nebula_dark;
        let mid = colors.nebula_mid;
        let bright = colors.nebula_bright;

        for y in 0..self.height {
            for x in 0..self.width {
                let nx = x as f32 / self.width as f32;
                let ny = y as f32 / self.height as f32;

                // Simple noise approximation
                let n1 = self.simple_noise(nx * 2.0, ny * 2.0, self.time);
                let n2 = self.simple_noise(nx * 5.0, ny * 5.0, self.time * 1.5);

                let combined = (n1 * 0.7 + n2 * 0.3) * (0.6 + audio.amplitude * 0.6);
                let combined = combined.clamp(0.0, 1.0);

                // Color blend
                let t = combined * 2.0;
                let t1 = t.min(1.0);
                let t2 = (t - 1.0).max(0.0).min(1.0);

                let r1 = dark[0] as f32 * (1.0 - t1) + mid[0] as f32 * t1;
                let g1 = dark[1] as f32 * (1.0 - t1) + mid[1] as f32 * t1;
                let b1 = dark[2] as f32 * (1.0 - t1) + mid[2] as f32 * t1;

                let r2 = mid[0] as f32 * (1.0 - t2) + bright[0] as f32 * t2;
                let g2 = mid[1] as f32 * (1.0 - t2) + bright[1] as f32 * t2;
                let b2 = mid[2] as f32 * (1.0 - t2) + bright[2] as f32 * t2;

                let (r, g, b) = if t > 1.0 { (r2, g2, b2) } else { (r1, g1, b1) };

                let r = (r * intensity + bg[0] as f32 * (1.0 - intensity * 0.5)) as u8;
                let g = (g * intensity + bg[1] as f32 * (1.0 - intensity * 0.5)) as u8;
                let b = (b * intensity + bg[2] as f32 * (1.0 - intensity * 0.5)) as u8;

                buffer.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
    }

    fn simple_noise(&self, x: f32, y: f32, t: f32) -> f32 {
        let n = (x * 127.1 + y * 311.7 + t).sin() * 43758.5453;
        n - n.floor()
    }
}
