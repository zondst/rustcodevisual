//! Particle Studio RS - Main Application
//! Full-featured music visualizer with egui GUI

mod config;
mod audio;
mod particles;
mod spectrum;
mod waveform;
mod postprocess;
mod compositor;
mod presets;
mod fractals;
mod export;
mod offscreen_render;
mod gpu_render;
mod gpu_export;

use eframe::egui;
use config::{AppConfig, ParticleMode, ParticleShape, SpectrumStyle, WaveformStyle, ColorScheme};
use audio::{AudioSystem, AudioState, AdaptiveAudioNormalizer, NormalizedAudio};
use particles::ParticleEngine;
use spectrum::SpectrumVisualizer;
use waveform::WaveformVisualizer;
use export::VideoExporter;
use offscreen_render::FrameRenderer;
use std::time::Instant;
use std::sync::mpsc::{self, Receiver};
use offscreen_render::ExportMessage;

/// Main application state
struct ParticleStudioApp {
    config: AppConfig,
    audio_sys: AudioSystem,
    audio_state: AudioState,
    particles: ParticleEngine,
    spectrum: SpectrumVisualizer,
    waveform: WaveformVisualizer,
    last_update: Instant,
    
    // UI state
    show_settings: bool,
    settings_tab: SettingsTab,
    current_frame: usize,
    is_playing: bool,
    last_dt: f32,
    selected_preset: usize,
    
    // Color scheme names
    color_scheme_names: Vec<String>,
    
    // Export state
    export_output_path: Option<std::path::PathBuf>,
    export_duration_secs: f32,
    export_is_exporting: bool,
    export_progress: f32,
    export_format: ExportFormat,
    export_error_message: Option<String>,
    export_current_frame: usize,
    export_total_frames: usize,
    ffmpeg_available: bool,
    video_exporter: VideoExporter,
    
    // Adaptive audio
    audio_normalizer: AdaptiveAudioNormalizer,
    normalized_audio: NormalizedAudio,
    
    // Frame renderer for export
    frame_renderer: FrameRenderer,
    
    // Export thread communication
    export_progress_rx: Option<Receiver<ExportMessage>>,

    // GPU Export state
    gpu_export_available: bool,
    gpu_info: Option<String>,
    use_gpu_export: bool,
    detected_encoder: gpu_export::HardwareEncoder,
    gpu_export_fps: f32,
    gpu_export_progress_rx: Option<std::sync::mpsc::Receiver<gpu_export::GpuExportMessage>>,
}



#[derive(Clone, Copy, PartialEq)]
enum ExportFormat {
    MP4,
    MOV,
}

impl Default for ExportFormat {
    fn default() -> Self { Self::MP4 }
}

#[derive(Clone, Copy, PartialEq)]
enum SettingsTab {
    Particles,
    Audio,
    Effects,
    Spectrum,
    Waveform,
    Export,
}

impl ParticleStudioApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Setup dark theme
        let mut visuals = egui::Visuals::dark();
        visuals.window_fill = egui::Color32::from_rgba_unmultiplied(15, 15, 25, 245);
        visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(20, 20, 35, 240);
        cc.egui_ctx.set_visuals(visuals);
        
        let config = AppConfig::default();
        let colors = config.get_color_scheme();
        
        let mut particles = ParticleEngine::new(1280.0, 720.0);
        particles.update_palette(&colors);
        
        let color_scheme_names: Vec<String> = ColorScheme::all_schemes()
            .iter()
            .map(|s| s.name.clone())
            .collect();
        
        Self {
            config,
            audio_sys: AudioSystem::new(),
            audio_state: AudioState::new(),
            particles,
            spectrum: SpectrumVisualizer::new(1280.0, 720.0),
            waveform: WaveformVisualizer::new(1280.0, 720.0),
            last_update: Instant::now(),
            show_settings: true,
            settings_tab: SettingsTab::Particles,
            current_frame: 0,
            is_playing: false,
            last_dt: 0.016,
            selected_preset: 0,
            color_scheme_names,
            // Export state
            export_output_path: None,
            export_duration_secs: 10.0,
            export_is_exporting: false,
            export_progress: 0.0,
            export_format: ExportFormat::MP4,
            export_error_message: None,
            export_current_frame: 0,
            export_total_frames: 0,
            ffmpeg_available: VideoExporter::check_ffmpeg_available(),
            video_exporter: VideoExporter::new(),
            // Adaptive audio
            audio_normalizer: AdaptiveAudioNormalizer::new(3.0, 30),
            normalized_audio: NormalizedAudio::default(),
            // Frame renderer for export
            frame_renderer: FrameRenderer::new(1920, 1080),
            // Export thread
            export_progress_rx: None,
            // GPU Export state
            gpu_export_available: gpu_export::is_gpu_export_available(),
            gpu_info: gpu_export::get_gpu_info(),
            use_gpu_export: true, // Default to GPU if available
            detected_encoder: gpu_export::HardwareEncoder::detect(),
            gpu_export_fps: 0.0,
            gpu_export_progress_rx: None,
        }
    }
    
    fn load_audio(&mut self, path: String) {
        let fps = self.config.export.fps;
        match self.audio_sys.load_file(path, fps) {
            Ok(()) => {
                self.current_frame = 0;
                self.is_playing = true;
                self.audio_sys.play();
            }
            Err(e) => {
                eprintln!("Error loading audio: {}", e);
            }
        }
    }
}

impl eframe::App for ParticleStudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        self.last_update = now;
        self.last_dt = dt;
        
        // Check for export thread progress updates (non-blocking)
        let mut should_clear_rx = false;
        if let Some(ref rx) = self.export_progress_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    ExportMessage::Progress(current, total) => {
                        self.export_current_frame = current;
                        self.export_total_frames = total;
                        self.export_progress = current as f32 / total as f32;
                    }
                    ExportMessage::Completed => {
                        self.export_is_exporting = false;
                        self.export_progress = 0.0;
                        should_clear_rx = true;
                        println!("Export completed!");
                    }
                    ExportMessage::Error(e) => {
                        self.export_error_message = Some(e);
                        self.export_is_exporting = false;
                        should_clear_rx = true;
                    }
                }
            }
        }
        if should_clear_rx {
            self.export_progress_rx = None;
        }

        // Check for GPU export progress updates
        let mut should_clear_gpu_rx = false;
        if let Some(ref rx) = self.gpu_export_progress_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    gpu_export::GpuExportMessage::Progress { current, total, fps } => {
                        self.export_current_frame = current;
                        self.export_total_frames = total;
                        self.export_progress = current as f32 / total as f32;
                        self.gpu_export_fps = fps;
                    }
                    gpu_export::GpuExportMessage::Completed { path } => {
                        self.export_is_exporting = false;
                        self.export_progress = 0.0;
                        should_clear_gpu_rx = true;
                        println!("GPU Export completed: {:?}", path);
                    }
                    gpu_export::GpuExportMessage::Error(e) => {
                        self.export_error_message = Some(e);
                        self.export_is_exporting = false;
                        should_clear_gpu_rx = true;
                    }
                }
            }
        }
        if should_clear_gpu_rx {
            self.gpu_export_progress_rx = None;
        }

        // Update audio state
        if let Some(ref analysis) = self.audio_sys.analysis {
            if self.is_playing && self.current_frame < analysis.total_frames {
                let frame = analysis.get_frame(self.current_frame);
                self.audio_state.update_from_frame(&frame, self.config.audio.smoothing);
                self.current_frame += 1;
            } else if self.current_frame >= analysis.total_frames {
                self.is_playing = false;
            }
        } else {
            // Fake audio for demo
            self.audio_state.update_fake(dt);
        }
        
        // Update smoothed audio values for gradual transitions
        self.audio_state.update_smoothing(
            dt, 
            self.config.audio.smoothing,
            self.config.audio.beat_attack,
            self.config.audio.beat_decay
        );
        
        // Update adaptive audio normalization
        if self.config.particles.adaptive_audio_enabled {
            self.normalized_audio = self.audio_normalizer.normalize(
                self.audio_state.smooth_bass,
                self.audio_state.smooth_mid,
                self.audio_state.smooth_high,
                self.config.particles.bass_sensitivity,
                self.config.particles.mid_sensitivity,
                self.config.particles.high_sensitivity,
                self.config.particles.adaptive_strength,
            );
        }
        
        // Update visualizers
        self.spectrum.update(&self.audio_state, &self.config.spectrum);
        self.waveform.update(&self.audio_state, &self.config.waveform);
        
        // UI Layout
        self.render_top_bar(ctx);
        
        if self.show_settings {
            self.render_settings_panel(ctx);
        }
        
        self.render_canvas(ctx, dt);
        
        // Request continuous repaint for animation
        ctx.request_repaint();
    }
}

impl ParticleStudioApp {
    fn render_top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🎵 Particle Studio RS");
                ui.separator();
                
                if ui.button("📂 Load Audio").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Audio", &["mp3", "wav", "ogg", "flac"])
                        .pick_file()
                    {
                        self.load_audio(path.to_string_lossy().to_string());
                    }
                }
                
                ui.separator();
                
                // Playback controls
                if self.audio_sys.analysis.is_some() {
                    if self.is_playing {
                        if ui.button("⏸ Pause").clicked() {
                            self.is_playing = false;
                            self.audio_sys.pause();
                        }
                    } else {
                        if ui.button("▶ Play").clicked() {
                            self.is_playing = true;
                            self.audio_sys.play();
                        }
                    }
                    
                    if ui.button("⏹ Stop").clicked() {
                        self.is_playing = false;
                        self.current_frame = 0;
                        self.audio_sys.stop();
                    }
                }
                
                ui.separator();
                
                // Settings toggle
                ui.toggle_value(&mut self.show_settings, "⚙ Settings");
                
                ui.separator();
                
                // Adaptive audio toggle (prominent on top bar)
                let adaptive_text = if self.config.particles.adaptive_audio_enabled {
                    "🎵 Adaptive ON"
                } else {
                    "🎵 Adaptive OFF"
                };
                if ui.selectable_label(self.config.particles.adaptive_audio_enabled, adaptive_text)
                    .on_hover_text("Automatically adjust to track dynamics")
                    .clicked() 
                {
                    self.config.particles.adaptive_audio_enabled = !self.config.particles.adaptive_audio_enabled;
                    if self.config.particles.adaptive_audio_enabled {
                        self.audio_normalizer.reset();
                    }
                }
                
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // FPS display
                    let fps = 1.0 / self.last_dt.max(0.001);
                    ui.label(format!("FPS: {:.0}", fps));
                    
                    // Audio info
                    if let Some(ref analysis) = self.audio_sys.analysis {
                        let progress = self.current_frame as f32 / analysis.total_frames as f32;
                        let current_time = self.current_frame as f32 / analysis.fps as f32;
                        ui.label(format!("{:.1}s / {:.1}s ({:.0}%)", 
                            current_time, analysis.duration, progress * 100.0));
                    }
                });
            });
        });
    }
    
    fn render_settings_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("settings_panel")
            .min_width(280.0)
            .show(ctx, |ui| {
                ui.heading("Settings");
                ui.separator();
                
                // Tab buttons
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.settings_tab, SettingsTab::Particles, "Particles");
                    ui.selectable_value(&mut self.settings_tab, SettingsTab::Effects, "Effects");
                    ui.selectable_value(&mut self.settings_tab, SettingsTab::Audio, "Audio");
                });
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.settings_tab, SettingsTab::Spectrum, "Spectrum");
                    ui.selectable_value(&mut self.settings_tab, SettingsTab::Waveform, "Waveform");
                    ui.selectable_value(&mut self.settings_tab, SettingsTab::Export, "Export");
                });
                
                ui.separator();
                
                // Preset selector
                ui.horizontal(|ui| {
                    ui.label("Preset:");
                    let preset_names = AppConfig::preset_names();
                    egui::ComboBox::from_id_source("preset_combo")
                        .selected_text(preset_names.get(self.selected_preset).copied().unwrap_or("Default"))
                        .show_ui(ui, |ui| {
                            for (i, name) in preset_names.iter().enumerate() {
                                if ui.selectable_value(&mut self.selected_preset, i, *name).changed() {
                                    self.config.apply_preset(name);
                                    self.particles.update_palette(&self.config.get_color_scheme());
                                }
                            }
                        });
                });
                
                ui.separator();
                
                egui::ScrollArea::vertical().show(ui, |ui| {
                    match self.settings_tab {
                        SettingsTab::Particles => self.render_particles_settings(ui),
                        SettingsTab::Audio => self.render_audio_settings(ui),
                        SettingsTab::Effects => self.render_effects_settings(ui),
                        SettingsTab::Spectrum => self.render_spectrum_settings(ui),
                        SettingsTab::Waveform => self.render_waveform_settings(ui),
                        SettingsTab::Export => self.render_export_settings(ui),
                    }
                });
            });
    }
    
    fn render_particles_settings(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.config.particles.enabled, "Enable Particles");
        
        ui.add_space(8.0);
        ui.label("Count");
        ui.add(egui::Slider::new(&mut self.config.particles.count, 1..=20000));  // Min 1 instead of 100
        
        ui.label("Speed");
        ui.add(egui::Slider::new(&mut self.config.particles.speed, 0.0..=10.0));  // Allow 0
        
        ui.label("Min Size");
        ui.add(egui::Slider::new(&mut self.config.particles.min_size, 0.5..=50.0));
        
        ui.label("Max Size");
        ui.add(egui::Slider::new(&mut self.config.particles.max_size, 0.5..=100.0));
        
        ui.label("Gravity");
        ui.add(egui::Slider::new(&mut self.config.particles.gravity, -2.0..=2.0));
        
        ui.label("Size Variation");
        ui.add(egui::Slider::new(&mut self.config.particles.size_variation, 0.0..=1.0));
        
        ui.label("Glow Intensity");
        ui.add(egui::Slider::new(&mut self.config.particles.glow_intensity, 0.0..=1.0));
        
        ui.label("Damping");
        ui.add(egui::Slider::new(&mut self.config.particles.damping, 0.0..=10.0));
        
        ui.add_space(8.0);
        ui.separator();
        ui.heading("Audio Reactivity");
        
        ui.checkbox(&mut self.config.particles.audio_reactive_spawn, "Audio-Reactive Spawn");
        ui.label("(Particles only appear when audio plays)");
        
        if self.config.particles.audio_reactive_spawn {
            ui.add_space(4.0);
            
            // Adaptive Audio section
            ui.horizontal(|ui| {
                if ui.checkbox(&mut self.config.particles.adaptive_audio_enabled, "🎵 Adaptive Audio").changed() {
                    if self.config.particles.adaptive_audio_enabled {
                        self.audio_normalizer.reset();
                    }
                }
                ui.label("(auto-adjust to track)");
            });
            
            if self.config.particles.adaptive_audio_enabled {
                ui.label("Strength (0=raw, 1=balanced)");
                ui.add(egui::Slider::new(&mut self.config.particles.adaptive_strength, 0.0..=2.0)
                    .suffix("")
                    .text(""));
                
                ui.add_space(4.0);
                ui.label("Window (secs)");
                ui.add(egui::Slider::new(&mut self.config.particles.adaptive_window_secs, 1.0..=5.0));
                
                ui.label("Bass Sensitivity");
                ui.add(egui::Slider::new(&mut self.config.particles.bass_sensitivity, 0.5..=2.0));
                
                ui.label("Mid Sensitivity");
                ui.add(egui::Slider::new(&mut self.config.particles.mid_sensitivity, 0.5..=2.0));
                
                ui.label("High Sensitivity");
                ui.add(egui::Slider::new(&mut self.config.particles.high_sensitivity, 0.3..=1.5));
            } else {
                ui.label("Spawn Threshold");
                ui.add(egui::Slider::new(&mut self.config.particles.audio_spawn_threshold, 0.0..=0.3));
            }
            
            ui.checkbox(&mut self.config.particles.fade_without_audio, "Fade Without Audio");
            
            ui.add_space(8.0);
            ui.label("Beat Burst Strength");
            ui.add(egui::Slider::new(&mut self.config.particles.beat_burst_strength, 0.0..=3.0));
            
            ui.label("Fade Attack (appear speed)");
            ui.add(egui::Slider::new(&mut self.config.particles.fade_attack_speed, 1.0..=10.0));
            
            ui.label("Fade Release (fade speed)");
            ui.add(egui::Slider::new(&mut self.config.particles.fade_release_speed, 0.5..=5.0));
            
            ui.add_space(8.0);
            ui.checkbox(&mut self.config.particles.spawn_from_center, "Spawn from Center");
            if self.config.particles.spawn_from_center {
                ui.label("Spawn Radius");
                ui.add(egui::Slider::new(&mut self.config.particles.spawn_radius, 10.0..=200.0));
            }
        }
        
        ui.add_space(8.0);
        ui.separator();
        ui.heading("Rendering");
        
        ui.checkbox(&mut self.config.particles.volumetric_rendering, "Volumetric (Smooth Gradients)");
        if self.config.particles.volumetric_rendering {
            ui.label("Gradient Steps");
            ui.add(egui::Slider::new(&mut self.config.particles.volumetric_steps, 8..=48));
        }
        
        ui.add_space(8.0);
        ui.label("Mode");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.config.particles.mode, ParticleMode::Chaos, "Chaos");
            ui.selectable_value(&mut self.config.particles.mode, ParticleMode::Calm, "Calm");
        });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.config.particles.mode, ParticleMode::Orbit, "Orbit");
            ui.selectable_value(&mut self.config.particles.mode, ParticleMode::Cinematic, "Cinematic");
        });
        
        ui.add_space(8.0);
        ui.label("Shape");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.config.particles.shape, ParticleShape::Circle, "Circle");
            ui.selectable_value(&mut self.config.particles.shape, ParticleShape::Glow, "Glow");
            ui.selectable_value(&mut self.config.particles.shape, ParticleShape::Ring, "Ring");
        });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.config.particles.shape, ParticleShape::Star, "Star");
            ui.selectable_value(&mut self.config.particles.shape, ParticleShape::Diamond, "Diamond");
            ui.selectable_value(&mut self.config.particles.shape, ParticleShape::Spark, "Spark");
        });
        
        ui.add_space(8.0);
        ui.label("Color Scheme");
        egui::ComboBox::from_label("")
            .selected_text(&self.color_scheme_names[self.config.color_scheme_index])
            .show_ui(ui, |ui| {
                for (i, name) in self.color_scheme_names.iter().enumerate() {
                    if ui.selectable_value(&mut self.config.color_scheme_index, i, name).changed() {
                        self.particles.update_palette(&self.config.get_color_scheme());
                    }
                }
            });
    }
    
    fn render_audio_settings(&mut self, ui: &mut egui::Ui) {
        ui.label("Smoothing");
        ui.add(egui::Slider::new(&mut self.config.audio.smoothing, 0.0..=0.99));
        
        ui.label("Beat Sensitivity");
        ui.add(egui::Slider::new(&mut self.config.audio.beat_sensitivity, 0.1..=2.0));
        
        ui.add_space(8.0);
        ui.label("Frequency Response");
        
        ui.label("Bass");
        ui.add(egui::Slider::new(&mut self.config.audio.bass_response, 0.0..=3.0));
        
        ui.label("Mid");
        ui.add(egui::Slider::new(&mut self.config.audio.mid_response, 0.0..=3.0));
        
        ui.label("High");
        ui.add(egui::Slider::new(&mut self.config.audio.high_response, 0.0..=3.0));
        
        ui.add_space(8.0);
        ui.label("Beat Explosion Strength");
        ui.add(egui::Slider::new(&mut self.config.audio.beat_explosion_strength, 0.0..=10.0));
        
        ui.label("Beat Size Pulse");
        ui.add(egui::Slider::new(&mut self.config.audio.beat_size_pulse, 0.0..=5.0));
    }
    
    fn render_effects_settings(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Bloom", |ui| {
            ui.checkbox(&mut self.config.visual.bloom_enabled, "Enable");
            ui.label("Intensity");
            ui.add(egui::Slider::new(&mut self.config.visual.bloom_intensity, 0.0..=2.0));
            ui.label("Radius");
            ui.add(egui::Slider::new(&mut self.config.visual.bloom_radius, 5.0..=50.0));
            ui.label("Threshold");
            ui.add(egui::Slider::new(&mut self.config.visual.bloom_threshold, 0.0..=1.0));
        });
        
        ui.collapsing("Motion & Vignette", |ui| {
            ui.label("Motion Blur");
            ui.add(egui::Slider::new(&mut self.config.visual.motion_blur, 0.0..=0.9));
            ui.label("Vignette");
            ui.add(egui::Slider::new(&mut self.config.visual.vignette_strength, 0.0..=1.0));
            ui.label("Film Grain");
            ui.add(egui::Slider::new(&mut self.config.visual.film_grain, 0.0..=0.2));
        });
        
        ui.collapsing("Color Effects", |ui| {
            ui.label("Chromatic Aberration");
            ui.add(egui::Slider::new(&mut self.config.visual.chromatic_aberration, 0.0..=1.0));
            ui.checkbox(&mut self.config.visual.scanlines, "Scanlines");
            if self.config.visual.scanlines {
                ui.add(egui::Slider::new(&mut self.config.visual.scanline_intensity, 0.0..=0.5));
            }
            ui.checkbox(&mut self.config.visual.color_shift_enabled, "Color Shift");
        });
        
        ui.collapsing("MilkDrop Effects", |ui| {
            ui.checkbox(&mut self.config.visual.echo_enabled, "Echo/Feedback");
            if self.config.visual.echo_enabled {
                ui.label("Zoom");
                ui.add(egui::Slider::new(&mut self.config.visual.echo_zoom, 1.0..=1.1));
                ui.label("Rotation");
                ui.add(egui::Slider::new(&mut self.config.visual.echo_rotation, -0.1..=0.1));
                ui.label("Alpha");
                ui.add(egui::Slider::new(&mut self.config.visual.echo_alpha, 0.0..=1.0));
            }
            
            ui.checkbox(&mut self.config.visual.kaleidoscope_enabled, "Kaleidoscope");
            if self.config.visual.kaleidoscope_enabled {
                let mut segments = self.config.visual.kaleidoscope_segments as i32;
                ui.add(egui::Slider::new(&mut segments, 3..=12).text("Segments"));
                self.config.visual.kaleidoscope_segments = segments as usize;
            }
            
            ui.checkbox(&mut self.config.visual.radial_blur_enabled, "Radial Blur");
            if self.config.visual.radial_blur_enabled {
                ui.add(egui::Slider::new(&mut self.config.visual.radial_blur_amount, 0.0..=0.5));
            }
        });
        
        ui.collapsing("Tone Mapping", |ui| {
            ui.label("Exposure");
            ui.add(egui::Slider::new(&mut self.config.visual.exposure, 0.5..=2.0));
            ui.label("Contrast");
            ui.add(egui::Slider::new(&mut self.config.visual.contrast, 0.5..=2.0));
            ui.label("Saturation");
            ui.add(egui::Slider::new(&mut self.config.visual.saturation, 0.0..=2.0));
            ui.label("Gamma");
            ui.add(egui::Slider::new(&mut self.config.visual.gamma, 0.5..=2.0));
        });
        
        ui.collapsing("UI Controls", |ui| {
            ui.checkbox(&mut self.config.visual.show_audio_meters, "Show Audio Meters");
            ui.label("(Bass/Mid/High bars at bottom-left)");
        });
    }
    
    fn render_spectrum_settings(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.config.spectrum.enabled, "Enable Spectrum");
        
        ui.add_space(8.0);
        ui.label("Style");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.config.spectrum.style, SpectrumStyle::Bars, "Bars");
            ui.selectable_value(&mut self.config.spectrum.style, SpectrumStyle::MirrorBars, "Mirror");
            ui.selectable_value(&mut self.config.spectrum.style, SpectrumStyle::Line, "Line");
        });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.config.spectrum.style, SpectrumStyle::Circle, "Circle");
            ui.selectable_value(&mut self.config.spectrum.style, SpectrumStyle::Waterfall, "Waterfall");
        });
        
        ui.add_space(8.0);
        let mut bar_count = self.config.spectrum.bar_count as i32;
        ui.label("Bar Count");
        ui.add(egui::Slider::new(&mut bar_count, 16..=128));
        self.config.spectrum.bar_count = bar_count as usize;
        
        ui.label("Height Scale");
        ui.add(egui::Slider::new(&mut self.config.spectrum.bar_height_scale, 0.1..=2.0));
        
        ui.label("Smoothing");
        ui.add(egui::Slider::new(&mut self.config.spectrum.smoothing, 0.0..=0.99));
        
        ui.label("Position Y");
        ui.add(egui::Slider::new(&mut self.config.spectrum.position_y, 0.0..=1.0));
    }
    
    fn render_waveform_settings(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.config.waveform.enabled, "Enable Waveform");
        
        ui.add_space(8.0);
        ui.label("Style");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.config.waveform.style, WaveformStyle::Line, "Line");
            ui.selectable_value(&mut self.config.waveform.style, WaveformStyle::Bars, "Bars");
        });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.config.waveform.style, WaveformStyle::Circle, "Circle");
            ui.selectable_value(&mut self.config.waveform.style, WaveformStyle::Mirror, "Mirror");
        });
        
        ui.add_space(8.0);
        ui.label("Thickness");
        ui.add(egui::Slider::new(&mut self.config.waveform.thickness, 1.0..=10.0));
        
        ui.label("Amplitude");
        ui.add(egui::Slider::new(&mut self.config.waveform.amplitude, 50.0..=400.0));
        
        ui.label("Position Y");
        ui.add(egui::Slider::new(&mut self.config.waveform.position_y, 0.0..=1.0));
        
        ui.checkbox(&mut self.config.waveform.mirror, "Mirror");
    }
    
    fn render_export_settings(&mut self, ui: &mut egui::Ui) {
        ui.label("Resolution");
        ui.horizontal(|ui| {
            let mut width = self.config.export.width as i32;
            let mut height = self.config.export.height as i32;
            ui.add(egui::DragValue::new(&mut width).prefix("W: ").clamp_range(640..=3840));
            ui.add(egui::DragValue::new(&mut height).prefix("H: ").clamp_range(360..=2160));
            self.config.export.width = width as u32;
            self.config.export.height = height as u32;
        });
        
        ui.add_space(8.0);
        ui.label("Presets");
        ui.horizontal(|ui| {
            if ui.button("720p").clicked() {
                self.config.export.width = 1280;
                self.config.export.height = 720;
            }
            if ui.button("1080p").clicked() {
                self.config.export.width = 1920;
                self.config.export.height = 1080;
            }
            if ui.button("4K").clicked() {
                self.config.export.width = 3840;
                self.config.export.height = 2160;
            }
        });
        
        ui.add_space(8.0);
        let mut fps = self.config.export.fps as i32;
        ui.label("FPS");
        ui.add(egui::Slider::new(&mut fps, 24..=60));
        self.config.export.fps = fps as u32;
        
        let mut crf = self.config.export.crf as i32;
        ui.label("Quality (CRF)");
        ui.add(egui::Slider::new(&mut crf, 15..=30));
        self.config.export.crf = crf as u32;
        
        // ================================================================
        // VIDEO EXPORT SECTION
        // ================================================================
        ui.add_space(16.0);
        ui.separator();
        ui.heading("🎬 Video Export");
        
        // Format selection
        ui.horizontal(|ui| {
            ui.label("Format:");
            ui.selectable_value(&mut self.export_format, ExportFormat::MP4, "MP4");
            ui.selectable_value(&mut self.export_format, ExportFormat::MOV, "MOV (Alpha)");
        });
        
        // Output file selection
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("Output:");
            if let Some(ref path) = self.export_output_path {
                let filename = path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "selected".to_string());
                ui.label(format!("📄 {}", filename));
            } else {
                ui.label("(not selected)");
            }
            
            if ui.button("📁 Select File").clicked() {
                let filter = match self.export_format {
                    ExportFormat::MP4 => ("MP4 Video", "mp4"),
                    ExportFormat::MOV => ("MOV Video", "mov"),
                };
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter(filter.0, &[filter.1])
                    .save_file()
                {
                    self.export_output_path = Some(path);
                }
            }
        });
        
        // Duration
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("Duration:");
            ui.add(egui::DragValue::new(&mut self.export_duration_secs)
                .suffix(" sec")
                .clamp_range(1.0..=600.0)
                .speed(0.5));
            
            if ui.button("10s").clicked() {
                self.export_duration_secs = 10.0;
            }
            if ui.button("30s").clicked() {
                self.export_duration_secs = 30.0;
            }
            if ui.button("Full").clicked() {
                if let Some(duration) = self.audio_sys.get_duration() {
                    self.export_duration_secs = duration;
                }
            }
        });
        
        // Export controls
        ui.add_space(12.0);

        // GPU Acceleration Section
        ui.heading("🚀 GPU Acceleration");
        if self.gpu_export_available {
            if let Some(ref gpu_name) = self.gpu_info {
                ui.label(format!("✅ GPU: {}", gpu_name));
            }
            ui.label(format!("Encoder: {}", self.detected_encoder.name()));
            ui.checkbox(&mut self.use_gpu_export, "Use GPU-accelerated export (3-5x faster)");
        } else {
            ui.colored_label(egui::Color32::YELLOW, "⚠ GPU not available, using CPU");
            self.use_gpu_export = false;
        }

        ui.add_space(8.0);

        // FFmpeg availability check
        if !self.ffmpeg_available {
            ui.colored_label(egui::Color32::YELLOW, "⚠ FFmpeg not found in PATH");
            ui.label("Install FFmpeg to enable video export");
            ui.hyperlink_to("Download FFmpeg", "https://ffmpeg.org/download.html");
        } else {
            ui.colored_label(egui::Color32::GREEN, "✓ FFmpeg available");
        }

        ui.add_space(8.0);

        if !self.export_is_exporting {
            let can_export = self.export_output_path.is_some()
                && self.audio_sys.is_loaded()
                && self.ffmpeg_available;

            if can_export {
                let button_text = if self.use_gpu_export && self.gpu_export_available {
                    "▶ Start GPU Export"
                } else {
                    "▶ Start Export (CPU)"
                };

                if ui.button(button_text).clicked() {
                    // Get required data for export
                    if let Some(ref path) = &self.export_output_path {
                        if let Some(ref analysis) = &self.audio_sys.analysis {
                            let fps = self.config.export.fps;
                            let duration = self.export_duration_secs;

                            // Calculate total frames for UI
                            self.export_total_frames = (duration * fps as f32) as usize;
                            self.export_current_frame = 0;
                            self.export_progress = 0.0;
                            self.export_error_message = None;
                            self.export_is_exporting = true;

                            if self.use_gpu_export && self.gpu_export_available {
                                // GPU Export path
                                let (tx, rx) = mpsc::channel();
                                self.gpu_export_progress_rx = Some(rx);

                                let export_cfg = gpu_export::GpuExportConfig {
                                    width: self.config.export.width,
                                    height: self.config.export.height,
                                    fps: self.config.export.fps,
                                    duration_secs: duration,
                                    output_path: path.clone(),
                                    audio_path: self.audio_sys.audio_path.clone().map(|s| std::path::PathBuf::from(s)),
                                    encoder: self.detected_encoder,
                                    quality: self.config.export.crf,
                                };


                                println!("=== GPU EXPORT DEBUG ===");
                                println!("GPU available: {}", self.gpu_export_available);
                                println!("GPU info: {:?}", self.gpu_info);
                                println!("Encoder: {:?}", self.detected_encoder.name());
                                println!("FFmpeg available: {}", self.ffmpeg_available);
                                println!("Export path: {:?}", path);
                                println!("Resolution: {}x{}", self.config.export.width, self.config.export.height);
                                println!("Duration: {} secs", duration);
                                println!("========================");

                                gpu_export::run_gpu_export(
                                    self.config.clone(),
                                    (**analysis).clone(),
                                    export_cfg,
                                    tx,
                                );

                                println!("GPU export started with {} encoder!", self.detected_encoder.name());
                            } else {
                                // CPU Export path (original headless)
                                let (tx, rx) = mpsc::channel();
                                self.export_progress_rx = Some(rx);

                                let config_clone = self.config.clone();
                                let analysis_clone = (**analysis).clone();
                                let path_clone = path.clone();

                                std::thread::spawn(move || {
                                    offscreen_render::run_headless_export(
                                        config_clone,
                                        analysis_clone,
                                        path_clone,
                                        duration,
                                        tx,
                                    );
                                });

                                println!("CPU export started in background thread!");
                            }
                        }
                    }
                }
            } else {
                ui.add_enabled(false, egui::Button::new("▶ Start Export"));
                if !self.audio_sys.is_loaded() {
                    ui.label("⚠ Load audio first");
                } else if self.export_output_path.is_none() {
                    ui.label("⚠ Select output file");
                } else if !self.ffmpeg_available {
                    ui.label("⚠ FFmpeg required");
                }
            }
            
            // Show error message if any
            if let Some(ref err) = self.export_error_message {
                ui.colored_label(egui::Color32::RED, format!("❌ {}", err));
            }
        } else {
            // Export in progress (running in background thread)
            let mode_str = if self.gpu_export_progress_rx.is_some() { "GPU" } else { "CPU" };
            let frame_info = format!("Frame {}/{} ({} Export)", self.export_current_frame, self.export_total_frames, mode_str);
            ui.label(frame_info);

            ui.add(egui::ProgressBar::new(self.export_progress)
                .show_percentage()
                .animate(true));

            // Show FPS for GPU export
            if self.gpu_export_progress_rx.is_some() && self.gpu_export_fps > 0.0 {
                let target_fps = self.config.export.fps as f32;
                let speed_ratio = self.gpu_export_fps / target_fps;
                ui.label(format!("🚀 {:.1} fps ({:.1}x realtime)", self.gpu_export_fps, speed_ratio));
            } else {
                ui.label("🔄 Rendering in background - UI stays responsive!");
            }

            // ETA calculation
            if self.export_current_frame > 0 && self.gpu_export_fps > 0.0 {
                let remaining_frames = self.export_total_frames - self.export_current_frame;
                let eta_secs = remaining_frames as f32 / self.gpu_export_fps;
                if eta_secs < 60.0 {
                    ui.label(format!("⏱ ETA: {:.0} seconds", eta_secs));
                } else {
                    ui.label(format!("⏱ ETA: {:.1} minutes", eta_secs / 60.0));
                }
            }
        }
        
        // ================================================================
        // CONFIG SAVE/LOAD
        // ================================================================
        ui.add_space(16.0);
        ui.separator();
        ui.heading("💾 Configuration");
        
        if ui.button("💾 Save Config").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("JSON", &["json"])
                .save_file()
            {
                if let Err(e) = self.config.save(&path.to_string_lossy()) {
                    eprintln!("Error saving config: {}", e);
                }
            }
        }
        
        if ui.button("📂 Load Config").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("JSON", &["json"])
                .pick_file()
            {
                match AppConfig::load(&path.to_string_lossy()) {
                    Ok(config) => {
                        self.config = config;
                        self.particles.update_palette(&self.config.get_color_scheme());
                    }
                    Err(e) => eprintln!("Error loading config: {}", e),
                }
            }
        }
    }
    
    fn render_canvas(&mut self, ctx: &egui::Context, dt: f32) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
            
            // Update engine dimensions
            self.particles.width = rect.width();
            self.particles.height = rect.height();
            self.spectrum.resize(rect.width(), rect.height());
            self.waveform.resize(rect.width(), rect.height());
            
            // Update particles
            let normalized_ref = if self.config.particles.adaptive_audio_enabled {
                Some(&self.normalized_audio)
            } else {
                None
            };
            self.particles.update(&self.config.particles, &self.audio_state, dt, normalized_ref);
            
            let painter = ui.painter_at(rect);
            let colors = self.config.get_color_scheme();
            
            // Draw background
            let bg_color = egui::Color32::from_rgb(
                colors.background[0],
                colors.background[1], 
                colors.background[2]
            );
            painter.rect_filled(rect, 0.0, bg_color);
            
            // Draw spectrum (behind particles)
            self.spectrum.render(&painter, rect, &self.config.spectrum, &colors, &self.audio_state);
            
            // Draw waveform
            self.waveform.render(&painter, rect, &self.config.waveform, &colors, &self.audio_state);
            
            // Draw particles
            self.particles.render(&painter, rect, &self.config.particles, &self.audio_state);
            
            // Audio level indicators (conditionally)
            if self.config.visual.show_audio_meters {
                self.draw_audio_meters(ui, rect, &painter);
            }
        });
    }
    
    fn draw_audio_meters(&self, ui: &egui::Ui, rect: egui::Rect, painter: &egui::Painter) {
        let meter_width = 8.0;
        let meter_height = 100.0;
        let margin = 10.0;
        
        // Bass meter (left)
        let bass_rect = egui::Rect::from_min_size(
            egui::Pos2::new(rect.left() + margin, rect.bottom() - margin - meter_height),
            egui::Vec2::new(meter_width, meter_height),
        );
        painter.rect_filled(bass_rect, 2.0, egui::Color32::from_rgba_unmultiplied(50, 50, 80, 100));
        
        let bass_fill_height = meter_height * self.audio_state.bass.min(1.0);
        let bass_fill_rect = egui::Rect::from_min_size(
            egui::Pos2::new(bass_rect.left(), bass_rect.bottom() - bass_fill_height),
            egui::Vec2::new(meter_width, bass_fill_height),
        );
        painter.rect_filled(bass_fill_rect, 2.0, egui::Color32::from_rgb(255, 100, 100));
        
        // Mid meter
        let mid_rect = egui::Rect::from_min_size(
            egui::Pos2::new(rect.left() + margin + meter_width + 4.0, rect.bottom() - margin - meter_height),
            egui::Vec2::new(meter_width, meter_height),
        );
        painter.rect_filled(mid_rect, 2.0, egui::Color32::from_rgba_unmultiplied(50, 50, 80, 100));
        
        let mid_fill_height = meter_height * self.audio_state.mid.min(1.0);
        let mid_fill_rect = egui::Rect::from_min_size(
            egui::Pos2::new(mid_rect.left(), mid_rect.bottom() - mid_fill_height),
            egui::Vec2::new(meter_width, mid_fill_height),
        );
        painter.rect_filled(mid_fill_rect, 2.0, egui::Color32::from_rgb(100, 255, 100));
        
        // High meter
        let high_rect = egui::Rect::from_min_size(
            egui::Pos2::new(rect.left() + margin + (meter_width + 4.0) * 2.0, rect.bottom() - margin - meter_height),
            egui::Vec2::new(meter_width, meter_height),
        );
        painter.rect_filled(high_rect, 2.0, egui::Color32::from_rgba_unmultiplied(50, 50, 80, 100));
        
        let high_fill_height = meter_height * self.audio_state.high.min(1.0);
        let high_fill_rect = egui::Rect::from_min_size(
            egui::Pos2::new(high_rect.left(), high_rect.bottom() - high_fill_height),
            egui::Vec2::new(meter_width, high_fill_height),
        );
        painter.rect_filled(high_fill_rect, 2.0, egui::Color32::from_rgb(100, 100, 255));
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 800.0])
            .with_title("Particle Studio RS")
            .with_min_inner_size([800.0, 600.0]),
        vsync: false, // Disable vsync for max FPS
        ..Default::default()
    };
    
    eframe::run_native(
        "Particle Studio RS",
        options,
        Box::new(|cc| Box::new(ParticleStudioApp::new(cc))),
    )
}