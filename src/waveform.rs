//! Waveform Visualizer for Particle Studio RS
//! Audio waveform visualization with multiple styles

use egui::{Color32, Painter, Pos2, Rect};
use crate::config::{WaveformConfig, WaveformStyle, ColorScheme};
use crate::audio::AudioState;

pub struct WaveformVisualizer {
    pub width: f32,
    pub height: f32,
    smoothed_waveform: Vec<f32>,
}

impl WaveformVisualizer {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            width,
            height,
            smoothed_waveform: vec![0.0; 512],
        }
    }
    
    pub fn resize(&mut self, width: f32, height: f32) {
        self.width = width;
        self.height = height;
    }
    
    /// Update smoothed waveform values
    pub fn update(&mut self, audio: &AudioState, config: &WaveformConfig) {
        let smoothing = (config.smoothing as f32 / 20.0).clamp(0.0, 0.9);
        
        if self.smoothed_waveform.len() != audio.waveform.len() {
            self.smoothed_waveform = vec![0.0; audio.waveform.len()];
        }
        
        for (i, &val) in audio.waveform.iter().enumerate() {
            self.smoothed_waveform[i] = self.smoothed_waveform[i] * smoothing 
                + val * (1.0 - smoothing);
        }
    }
    
    /// Render waveform visualization
    pub fn render(
        &self,
        painter: &Painter,
        rect: Rect,
        config: &WaveformConfig,
        colors: &ColorScheme,
        audio: &AudioState,
    ) {
        if !config.enabled {
            return;
        }
        
        match config.style {
            WaveformStyle::Line => self.render_line(painter, rect, config, colors, audio),
            WaveformStyle::Bars => self.render_bars(painter, rect, config, colors, audio),
            WaveformStyle::Circle => self.render_circle(painter, rect, config, colors, audio),
            WaveformStyle::Mirror => self.render_mirror(painter, rect, config, colors, audio),
        }
    }
    
    fn render_line(
        &self,
        painter: &Painter,
        rect: Rect,
        config: &WaveformConfig,
        colors: &ColorScheme,
        audio: &AudioState,
    ) {
        let center_y = rect.top() + rect.height() * config.position_y;
        let amplitude = config.amplitude * (1.0 + audio.amplitude * 0.5);
        
        let wave_rgb = if config.use_scheme_color { colors.waveform } else { config.color };
        let color = Color32::from_rgb(wave_rgb[0], wave_rgb[1], wave_rgb[2]);
        
        let samples = self.smoothed_waveform.len();
        let mut points: Vec<Pos2> = Vec::with_capacity(samples);
        
        for (i, &value) in self.smoothed_waveform.iter().enumerate() {
            let x = rect.left() + (i as f32 / samples as f32) * rect.width();
            let y = center_y - value * amplitude;
            points.push(Pos2::new(x, y));
        }
        
        // Draw line with thickness
        let stroke = egui::Stroke::new(config.thickness, color);
        
        for i in 0..points.len().saturating_sub(1) {
            painter.line_segment([points[i], points[i + 1]], stroke);
        }
        
        // Glow effect
        let glow_color = Color32::from_rgba_unmultiplied(
            wave_rgb[0], wave_rgb[1], wave_rgb[2],
            (audio.amplitude * 80.0) as u8,
        );
        let glow_stroke = egui::Stroke::new(config.thickness * 3.0, glow_color);
        
        for i in 0..points.len().saturating_sub(1) {
            painter.line_segment([points[i], points[i + 1]], glow_stroke);
        }
    }
    
    fn render_bars(
        &self,
        painter: &Painter,
        rect: Rect,
        config: &WaveformConfig,
        colors: &ColorScheme,
        audio: &AudioState,
    ) {
        let center_y = rect.top() + rect.height() * config.position_y;
        let amplitude = config.amplitude * (1.0 + audio.amplitude * 0.5);
        
        let wave_rgb = if config.use_scheme_color { colors.waveform } else { config.color };
        let color = Color32::from_rgb(wave_rgb[0], wave_rgb[1], wave_rgb[2]);
        
        // Downsample for bar display
        let bar_count = 64;
        let samples_per_bar = self.smoothed_waveform.len() / bar_count;
        let bar_width = rect.width() / bar_count as f32 * 0.8;
        let bar_spacing = rect.width() / bar_count as f32 * 0.2 / 2.0;
        
        for i in 0..bar_count {
            let start_idx = i * samples_per_bar;
            let end_idx = ((i + 1) * samples_per_bar).min(self.smoothed_waveform.len());
            
            // Get max value in this segment
            let max_value = self.smoothed_waveform[start_idx..end_idx]
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);
            
            let height = max_value * amplitude;
            let x = rect.left() + i as f32 * (bar_width + bar_spacing * 2.0) + bar_spacing;
            
            // Upper bar
            let bar_rect = Rect::from_min_max(
                Pos2::new(x, center_y - height),
                Pos2::new(x + bar_width, center_y),
            );
            painter.rect_filled(bar_rect, 2.0, color);
            
            // Lower bar (mirror)
            if config.mirror {
                let mirror_rect = Rect::from_min_max(
                    Pos2::new(x, center_y),
                    Pos2::new(x + bar_width, center_y + height),
                );
                painter.rect_filled(mirror_rect, 2.0, color);
            }
        }
    }
    
    fn render_circle(
        &self,
        painter: &Painter,
        rect: Rect,
        config: &WaveformConfig,
        colors: &ColorScheme,
        audio: &AudioState,
    ) {
        let center = Pos2::new(
            rect.left() + rect.width() * config.position_x,
            rect.top() + rect.height() * config.position_y,
        );
        
        let base_radius = rect.width().min(rect.height()) * config.circular_radius;
        let amplitude = config.amplitude * 0.5 * (1.0 + audio.amplitude * 0.5);
        
        let wave_rgb = if config.use_scheme_color { colors.waveform } else { config.color };
        let color = Color32::from_rgb(wave_rgb[0], wave_rgb[1], wave_rgb[2]);
        
        let samples = self.smoothed_waveform.len();
        let angle_step = std::f32::consts::TAU / samples as f32;
        
        let mut points: Vec<Pos2> = Vec::with_capacity(samples + 1);
        
        for (i, &value) in self.smoothed_waveform.iter().enumerate() {
            let angle = i as f32 * angle_step - std::f32::consts::FRAC_PI_2;
            let radius = base_radius + value * amplitude;
            
            let x = center.x + angle.cos() * radius;
            let y = center.y + angle.sin() * radius;
            points.push(Pos2::new(x, y));
        }
        
        // Close the circle
        if let Some(&first) = points.first() {
            points.push(first);
        }
        
        // Draw waveform circle
        let stroke = egui::Stroke::new(config.thickness, color);
        for i in 0..points.len().saturating_sub(1) {
            painter.line_segment([points[i], points[i + 1]], stroke);
        }
        
        // Draw inner circle
        let inner_color = Color32::from_rgba_unmultiplied(
            wave_rgb[0], wave_rgb[1], wave_rgb[2],
            50,
        );
        painter.circle_stroke(center, base_radius * 0.8, egui::Stroke::new(1.0, inner_color));
    }
    
    fn render_mirror(
        &self,
        painter: &Painter,
        rect: Rect,
        config: &WaveformConfig,
        colors: &ColorScheme,
        audio: &AudioState,
    ) {
        let center_y = rect.top() + rect.height() * config.position_y;
        let amplitude = config.amplitude * (1.0 + audio.amplitude * 0.5);
        
        let wave_rgb = if config.use_scheme_color { colors.waveform } else { config.color };
        let color = Color32::from_rgb(wave_rgb[0], wave_rgb[1], wave_rgb[2]);
        
        let samples = self.smoothed_waveform.len();
        let mut upper_points: Vec<Pos2> = Vec::with_capacity(samples);
        let mut lower_points: Vec<Pos2> = Vec::with_capacity(samples);
        
        for (i, &value) in self.smoothed_waveform.iter().enumerate() {
            let x = rect.left() + (i as f32 / samples as f32) * rect.width();
            let offset = value.abs() * amplitude;
            
            upper_points.push(Pos2::new(x, center_y - offset));
            lower_points.push(Pos2::new(x, center_y + offset));
        }
        
        let stroke = egui::Stroke::new(config.thickness, color);
        
        // Draw upper line
        for i in 0..upper_points.len().saturating_sub(1) {
            painter.line_segment([upper_points[i], upper_points[i + 1]], stroke);
        }
        
        // Draw lower line
        for i in 0..lower_points.len().saturating_sub(1) {
            painter.line_segment([lower_points[i], lower_points[i + 1]], stroke);
        }
        
        // Fill between with transparent color
        let fill_color = Color32::from_rgba_unmultiplied(
            wave_rgb[0], wave_rgb[1], wave_rgb[2],
            30,
        );
        
        // Draw filled area using thin rectangles
        for i in 0..upper_points.len().saturating_sub(1) {
            let fill_rect = Rect::from_min_max(
                Pos2::new(upper_points[i].x, upper_points[i].y),
                Pos2::new(upper_points[i + 1].x, lower_points[i].y),
            );
            painter.rect_filled(fill_rect, 0.0, fill_color);
        }
        
        // Center line
        let center_color = Color32::from_rgba_unmultiplied(
            wave_rgb[0], wave_rgb[1], wave_rgb[2],
            100,
        );
        painter.line_segment(
            [Pos2::new(rect.left(), center_y), Pos2::new(rect.right(), center_y)],
            egui::Stroke::new(1.0, center_color),
        );
    }
}
