//! Spectrum Visualizer for Particle Studio RS
//! Audio spectrum visualization with multiple styles

use egui::{Color32, Painter, Pos2, Rect, Vec2};
use crate::config::{SpectrumConfig, SpectrumStyle, ColorScheme};
use crate::audio::AudioState;

pub struct SpectrumVisualizer {
    pub width: f32,
    pub height: f32,
    smoothed_spectrum: Vec<f32>,
    waterfall_history: Vec<Vec<f32>>,
    waterfall_max_rows: usize,
}

impl SpectrumVisualizer {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            width,
            height,
            smoothed_spectrum: vec![0.0; 64],
            waterfall_history: Vec::new(),
            waterfall_max_rows: 100,
        }
    }
    
    pub fn resize(&mut self, width: f32, height: f32) {
        self.width = width;
        self.height = height;
    }
    
    /// Update smoothed spectrum values
    pub fn update(&mut self, audio: &AudioState, config: &SpectrumConfig) {
        let smoothing = config.smoothing.clamp(0.0, 0.99);
        
        // Ensure same size
        if self.smoothed_spectrum.len() != audio.spectrum.len() {
            self.smoothed_spectrum = vec![0.0; audio.spectrum.len()];
        }
        
        // Apply smoothing
        for (i, &val) in audio.spectrum.iter().enumerate() {
            self.smoothed_spectrum[i] = self.smoothed_spectrum[i] * smoothing 
                + val * (1.0 - smoothing);
        }
        
        // Update waterfall history
        if config.style == SpectrumStyle::Waterfall {
            self.waterfall_history.insert(0, self.smoothed_spectrum.clone());
            if self.waterfall_history.len() > self.waterfall_max_rows {
                self.waterfall_history.pop();
            }
        }
    }
    
    /// Render spectrum visualization
    pub fn render(
        &self, 
        painter: &Painter, 
        rect: Rect,
        config: &SpectrumConfig,
        colors: &ColorScheme,
        audio: &AudioState,
    ) {
        if !config.enabled {
            return;
        }
        
        match config.style {
            SpectrumStyle::Bars => self.render_bars(painter, rect, config, colors, audio),
            SpectrumStyle::MirrorBars => self.render_mirror_bars(painter, rect, config, colors, audio),
            SpectrumStyle::Line => self.render_line(painter, rect, config, colors, audio),
            SpectrumStyle::Circle => self.render_circle(painter, rect, config, colors, audio),
            SpectrumStyle::Waterfall => self.render_waterfall(painter, rect, config, colors),
        }
    }
    
    fn render_bars(
        &self,
        painter: &Painter,
        rect: Rect,
        config: &SpectrumConfig,
        colors: &ColorScheme,
        audio: &AudioState,
    ) {
        let bar_count = config.bar_count.min(self.smoothed_spectrum.len());
        let total_width = rect.width();
        let bar_total_width = total_width / bar_count as f32;
        let bar_width = bar_total_width * config.bar_width;
        let bar_spacing = bar_total_width * config.bar_spacing / 2.0;
        
        let base_y = rect.bottom() - (rect.height() * config.position_y);
        let max_height = rect.height() * 0.8 * config.bar_height_scale;
        
        let color_low = Color32::from_rgb(colors.spectrum_low[0], colors.spectrum_low[1], colors.spectrum_low[2]);
        let color_high = Color32::from_rgb(colors.spectrum_high[0], colors.spectrum_high[1], colors.spectrum_high[2]);
        
        for i in 0..bar_count {
            let spectrum_idx = i * self.smoothed_spectrum.len() / bar_count;
            let value = self.smoothed_spectrum.get(spectrum_idx).copied().unwrap_or(0.0);
            
            // Apply beat boost
            let boosted_value = (value * (1.0 + audio.beat * 0.5)).min(1.0);
            let height = boosted_value * max_height;
            
            let x = rect.left() + i as f32 * bar_total_width + bar_spacing;
            
            // Gradient color based on frequency
            let t = i as f32 / bar_count as f32;
            let color = lerp_color(color_low, color_high, t);
            
            // Draw bar
            let bar_rect = Rect::from_min_max(
                Pos2::new(x, base_y - height),
                Pos2::new(x + bar_width, base_y),
            );
            
            painter.rect_filled(bar_rect, 2.0, color);
            
            // Add glow effect on top
            let glow_color = Color32::from_rgba_unmultiplied(
                color.r(), color.g(), color.b(), 
                (boosted_value * 100.0) as u8
            );
            let glow_rect = Rect::from_min_max(
                Pos2::new(x - 2.0, base_y - height - 4.0),
                Pos2::new(x + bar_width + 2.0, base_y - height + 4.0),
            );
            painter.rect_filled(glow_rect, 4.0, glow_color);
        }
    }
    
    fn render_mirror_bars(
        &self,
        painter: &Painter,
        rect: Rect,
        config: &SpectrumConfig,
        colors: &ColorScheme,
        audio: &AudioState,
    ) {
        let bar_count = config.bar_count.min(self.smoothed_spectrum.len());
        let total_width = rect.width();
        let bar_total_width = total_width / bar_count as f32;
        let bar_width = bar_total_width * config.bar_width;
        let bar_spacing = bar_total_width * config.bar_spacing / 2.0;
        
        let center_y = rect.center().y;
        let max_height = rect.height() * 0.4 * config.bar_height_scale;
        
        let color_low = Color32::from_rgb(colors.spectrum_low[0], colors.spectrum_low[1], colors.spectrum_low[2]);
        let color_high = Color32::from_rgb(colors.spectrum_high[0], colors.spectrum_high[1], colors.spectrum_high[2]);
        
        for i in 0..bar_count {
            let spectrum_idx = i * self.smoothed_spectrum.len() / bar_count;
            let value = self.smoothed_spectrum.get(spectrum_idx).copied().unwrap_or(0.0);
            
            let boosted_value = (value * (1.0 + audio.beat * 0.5)).min(1.0);
            let height = boosted_value * max_height;
            
            let x = rect.left() + i as f32 * bar_total_width + bar_spacing;
            
            let t = i as f32 / bar_count as f32;
            let color = lerp_color(color_low, color_high, t);
            
            // Upper bar
            let upper_rect = Rect::from_min_max(
                Pos2::new(x, center_y - height),
                Pos2::new(x + bar_width, center_y),
            );
            painter.rect_filled(upper_rect, 2.0, color);
            
            // Lower bar (mirrored)
            let lower_rect = Rect::from_min_max(
                Pos2::new(x, center_y),
                Pos2::new(x + bar_width, center_y + height),
            );
            painter.rect_filled(lower_rect, 2.0, color);
        }
    }
    
    fn render_line(
        &self,
        painter: &Painter,
        rect: Rect,
        config: &SpectrumConfig,
        colors: &ColorScheme,
        audio: &AudioState,
    ) {
        let bar_count = config.bar_count.min(self.smoothed_spectrum.len());
        let base_y = rect.bottom() - (rect.height() * config.position_y);
        let max_height = rect.height() * 0.8 * config.bar_height_scale;
        
        let color_low = Color32::from_rgb(colors.spectrum_low[0], colors.spectrum_low[1], colors.spectrum_low[2]);
        let color_high = Color32::from_rgb(colors.spectrum_high[0], colors.spectrum_high[1], colors.spectrum_high[2]);
        
        let mut points: Vec<Pos2> = Vec::with_capacity(bar_count);
        
        for i in 0..bar_count {
            let spectrum_idx = i * self.smoothed_spectrum.len() / bar_count;
            let value = self.smoothed_spectrum.get(spectrum_idx).copied().unwrap_or(0.0);
            
            let boosted_value = (value * (1.0 + audio.beat * 0.5)).min(1.0);
            let height = boosted_value * max_height;
            
            let x = rect.left() + (i as f32 / bar_count as f32) * rect.width();
            let y = base_y - height;
            
            points.push(Pos2::new(x, y));
        }
        
        // Draw line segments with gradient
        for i in 0..points.len().saturating_sub(1) {
            let t = i as f32 / points.len() as f32;
            let color = lerp_color(color_low, color_high, t);
            
            painter.line_segment(
                [points[i], points[i + 1]],
                egui::Stroke::new(2.0, color),
            );
        }
    }
    
    fn render_circle(
        &self,
        painter: &Painter,
        rect: Rect,
        config: &SpectrumConfig,
        colors: &ColorScheme,
        audio: &AudioState,
    ) {
        let center = rect.center();
        let base_radius = rect.width().min(rect.height()) * 0.25;
        let max_amplitude = rect.width().min(rect.height()) * 0.2 * config.bar_height_scale;
        
        let bar_count = config.bar_count.min(self.smoothed_spectrum.len());
        let angle_step = std::f32::consts::TAU / bar_count as f32;
        
        let color_low = Color32::from_rgb(colors.spectrum_low[0], colors.spectrum_low[1], colors.spectrum_low[2]);
        let color_high = Color32::from_rgb(colors.spectrum_high[0], colors.spectrum_high[1], colors.spectrum_high[2]);
        
        for i in 0..bar_count {
            let spectrum_idx = i * self.smoothed_spectrum.len() / bar_count;
            let value = self.smoothed_spectrum.get(spectrum_idx).copied().unwrap_or(0.0);
            
            let boosted_value = (value * (1.0 + audio.beat * 0.5)).min(1.0);
            let radius = base_radius + boosted_value * max_amplitude;
            
            let angle = i as f32 * angle_step - std::f32::consts::FRAC_PI_2;
            let inner_x = center.x + angle.cos() * base_radius;
            let inner_y = center.y + angle.sin() * base_radius;
            let outer_x = center.x + angle.cos() * radius;
            let outer_y = center.y + angle.sin() * radius;
            
            let t = i as f32 / bar_count as f32;
            let color = lerp_color(color_low, color_high, t);
            
            painter.line_segment(
                [Pos2::new(inner_x, inner_y), Pos2::new(outer_x, outer_y)],
                egui::Stroke::new(3.0, color),
            );
        }
        
        // Draw center circle
        let center_color = Color32::from_rgba_unmultiplied(
            colors.spectrum_low[0], 
            colors.spectrum_low[1], 
            colors.spectrum_low[2],
            100,
        );
        painter.circle_stroke(center, base_radius, egui::Stroke::new(2.0, center_color));
    }
    
    fn render_waterfall(
        &self,
        painter: &Painter,
        rect: Rect,
        config: &SpectrumConfig,
        colors: &ColorScheme,
    ) {
        let bar_count = config.bar_count.min(self.smoothed_spectrum.len());
        let row_height = rect.height() / self.waterfall_max_rows as f32;
        let col_width = rect.width() / bar_count as f32;
        
        for (row_idx, spectrum) in self.waterfall_history.iter().enumerate() {
            let y = rect.top() + row_idx as f32 * row_height;
            let alpha = 1.0 - (row_idx as f32 / self.waterfall_max_rows as f32);
            
            for i in 0..bar_count {
                let spectrum_idx = i * spectrum.len() / bar_count;
                let value = spectrum.get(spectrum_idx).copied().unwrap_or(0.0);
                
                if value < 0.01 {
                    continue;
                }
                
                let x = rect.left() + i as f32 * col_width;
                
                // Color based on intensity
                let t = value.min(1.0);
                let color = lerp_color(
                    Color32::from_rgb(colors.spectrum_low[0], colors.spectrum_low[1], colors.spectrum_low[2]),
                    Color32::from_rgb(colors.spectrum_high[0], colors.spectrum_high[1], colors.spectrum_high[2]),
                    t,
                );
                
                let final_color = Color32::from_rgba_unmultiplied(
                    color.r(), color.g(), color.b(),
                    (alpha * value * 255.0) as u8,
                );
                
                let cell_rect = Rect::from_min_size(
                    Pos2::new(x, y),
                    Vec2::new(col_width, row_height),
                );
                
                painter.rect_filled(cell_rect, 0.0, final_color);
            }
        }
    }
}

/// Linearly interpolate between two colors
fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgb(
        (a.r() as f32 * (1.0 - t) + b.r() as f32 * t) as u8,
        (a.g() as f32 * (1.0 - t) + b.g() as f32 * t) as u8,
        (a.b() as f32 * (1.0 - t) + b.b() as f32 * t) as u8,
    )
}
