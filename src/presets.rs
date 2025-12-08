use serde::{Deserialize, Serialize};
use crate::config::{ParticleMode, ParticleShape, SpectrumStyle, WaveformStyle};

/// Available preset types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresetType {
    /// Default balanced preset
    Default,
    /// Particles GPU preset (from rust-particles-main)
    ParticlesGPU,
    /// Raymarched fractal backgrounds with audio reactivity (from fractal_sugar)
    AudioFractals,
    /// VJ-style reactive visualization (from visualizer2-canon)
    VJReactive,
    /// Intensive bloom effects (from WebGPU-Bloom)
    BloomIntensive,
    /// Custom user configuration
    Custom,
}

impl Default for PresetType {
    fn default() -> Self {
        Self::Default
    }
}

impl PresetType {
    pub fn all() -> Vec<PresetType> {
        vec![
            Self::Default,
            Self::ParticlesGPU,
            Self::AudioFractals,
            Self::VJReactive,
            Self::BloomIntensive,
            Self::Custom,
        ]
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            Self::Default => "Default",
            Self::ParticlesGPU => "Particles GPU",
            Self::AudioFractals => "Audio Fractals",
            Self::VJReactive => "VJ Reactive",
            Self::BloomIntensive => "Bloom Intensive",
            Self::Custom => "Custom",
        }
    }
    
    pub fn description(&self) -> &'static str {
        match self {
            Self::Default => "Balanced visualization with particles and effects",
            Self::ParticlesGPU => "GPU compute-style simulation with attractors and heavy bloom",
            Self::AudioFractals => "Fractal backgrounds with audio-reactive coloring",
            Self::VJReactive => "VJ-style FFT spectrum with beat detection",
            Self::BloomIntensive => "Intense multi-pass bloom with emissive particles",
            Self::Custom => "Custom user configuration",
        }
    }
}

/// Complete preset configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct PresetConfig {
    pub preset_type: PresetType,
    pub particles: ParticlePreset,
    pub visuals: VisualsPreset,
    pub audio: AudioPreset,
    pub fractals: FractalPreset,
}

impl Default for PresetConfig {
    fn default() -> Self {
        Self::preset_default()
    }
}

/// Particle configuration for presets
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ParticlePreset {
    pub count: usize,
    pub min_size: f32,
    pub max_size: f32,
    pub speed: f32,
    pub mode: ParticleMode,
    pub shape: ParticleShape,
    pub glow_intensity: f32,
    pub damping: f32,
    pub beat_pulse: f32,
    pub attractor_strength: f32,
    pub curl_strength: f32,
    pub spring_mode: bool,
    pub use_3d: bool,
}

impl Default for ParticlePreset {
    fn default() -> Self {
        Self {
            count: 1500,
            min_size: 2.0,
            max_size: 6.0,
            speed: 0.8,
            mode: ParticleMode::Chaos,
            shape: ParticleShape::Circle,
            glow_intensity: 0.6,
            damping: 2.0,
            beat_pulse: 1.5,
            attractor_strength: 0.3,
            curl_strength: 0.6,
            spring_mode: false,
            use_3d: false,
        }
    }
}

/// Visual effects configuration for presets
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct VisualsPreset {
    pub bloom_enabled: bool,
    pub bloom_intensity: f32,
    pub bloom_threshold: f32,
    pub bloom_radius: f32,
    pub bloom_mip_levels: u32,
    pub motion_blur: f32,
    pub vignette: f32,
    pub chromatic_aberration: f32,
    pub echo_enabled: bool,
    pub echo_zoom: f32,
    pub echo_rotation: f32,
    pub kaleidoscope_enabled: bool,
    pub kaleidoscope_segments: usize,
    pub kaleidoscope_animated: bool,
}

impl Default for VisualsPreset {
    fn default() -> Self {
        Self {
            bloom_enabled: true,
            bloom_intensity: 0.6,
            bloom_threshold: 0.5,
            bloom_radius: 25.0,
            bloom_mip_levels: 5,
            motion_blur: 0.2,
            vignette: 0.3,
            chromatic_aberration: 0.0,
            echo_enabled: false,
            echo_zoom: 1.02,
            echo_rotation: 0.01,
            kaleidoscope_enabled: false,
            kaleidoscope_segments: 6,
            kaleidoscope_animated: false,
        }
    }
}

/// Audio reactivity configuration for presets
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct AudioPreset {
    pub smoothing: f32,
    pub beat_sensitivity: f32,
    pub beat_attack: f32,
    pub beat_decay: f32,
    pub bass_response: f32,
    pub mid_response: f32,
    pub high_response: f32,
    pub spectrum_enabled: bool,
    pub spectrum_style: SpectrumStyle,
    pub waveform_enabled: bool,
    pub waveform_style: WaveformStyle,
}

impl Default for AudioPreset {
    fn default() -> Self {
        Self {
            smoothing: 0.5,
            beat_sensitivity: 0.5,
            beat_attack: 0.6,
            beat_decay: 0.25,
            bass_response: 1.5,
            mid_response: 1.0,
            high_response: 0.8,
            spectrum_enabled: true,
            spectrum_style: SpectrumStyle::Bars,
            waveform_enabled: true,
            waveform_style: WaveformStyle::Line,
        }
    }
}

/// Fractal rendering configuration for presets
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct FractalPreset {
    pub enabled: bool,
    pub fractal_type: FractalType,
    pub iterations: u32,
    pub reactive_bass: [f32; 3],
    pub reactive_mids: [f32; 3],
    pub reactive_high: [f32; 3],
    pub camera_rotation_speed: f32,
    pub orbit_distance: f32,
}

impl Default for FractalPreset {
    fn default() -> Self {
        Self {
            enabled: false,
            fractal_type: FractalType::None,
            iterations: 100,
            reactive_bass: [0.0, 0.0, 0.0],
            reactive_mids: [0.0, 0.0, 0.0],
            reactive_high: [0.0, 0.0, 0.0],
            camera_rotation_speed: 0.02,
            orbit_distance: 2.5,
        }
    }
}

/// Fractal types from fractal_sugar
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum FractalType {
    /// No fractal (particles only)
    None,
    /// Mandelbox - Box-folding with sphere inversion
    Mandelbox,
    /// Mandelbulb - 3D Mandelbrot extension
    Mandelbulb,
    /// Klein bottle inspired IFS
    Klein,
    /// Menger sponge - Recursive cube removal
    MengerSponge,
    /// Sierpinski tetrahedron
    Sierpinski,
    /// Quaternion Julia set
    QuaternionJulia,
}

impl Default for FractalType {
    fn default() -> Self {
        Self::None
    }
}

impl FractalType {
    pub fn all() -> Vec<FractalType> {
        vec![
            Self::None,
            Self::Mandelbox,
            Self::Mandelbulb,
            Self::Klein,
            Self::MengerSponge,
            Self::Sierpinski,
            Self::QuaternionJulia,
        ]
    }
    
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Mandelbox => "Mandelbox",
            Self::Mandelbulb => "Mandelbulb",
            Self::Klein => "Klein",
            Self::MengerSponge => "Menger Sponge",
            Self::Sierpinski => "Sierpinski",
            Self::QuaternionJulia => "Quaternion Julia",
        }
    }
    
    pub fn id(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Mandelbox => 1,
            Self::Mandelbulb => 2,
            Self::Klein => 3,
            Self::MengerSponge => 4,
            Self::Sierpinski => 5,
            Self::QuaternionJulia => 6,
        }
    }
}

impl PresetConfig {
    /// Default balanced preset
    pub fn preset_default() -> Self {
        Self {
            preset_type: PresetType::Default,
            particles: ParticlePreset::default(),
            visuals: VisualsPreset::default(),
            audio: AudioPreset::default(),
            fractals: FractalPreset::default(),
        }
    }
    
    /// Particles GPU preset (from rust-particles-main)
    /// Heavy particle simulation with attractors and bloom
    pub fn preset_particles_gpu() -> Self {
        Self {
            preset_type: PresetType::ParticlesGPU,
            particles: ParticlePreset {
                count: 5000,
                min_size: 2.0,
                max_size: 6.0,
                speed: 1.5,
                mode: ParticleMode::Chaos,
                shape: ParticleShape::Glow,
                glow_intensity: 0.8,
                damping: 1.5,
                beat_pulse: 2.5,
                attractor_strength: 0.5,
                curl_strength: 0.8,
                spring_mode: false,
                use_3d: false,
            },
            visuals: VisualsPreset {
                bloom_enabled: true,
                bloom_intensity: 1.8,
                bloom_threshold: 0.2,
                bloom_radius: 45.0,
                bloom_mip_levels: 7,
                motion_blur: 0.2,
                vignette: 0.3,
                chromatic_aberration: 0.0,
                echo_enabled: false,
                echo_zoom: 1.0,
                echo_rotation: 0.0,
                kaleidoscope_enabled: false,
                kaleidoscope_segments: 6,
                kaleidoscope_animated: false,
            },
            audio: AudioPreset {
                smoothing: 0.3,
                beat_sensitivity: 0.6,
                beat_attack: 0.8,
                beat_decay: 0.35,
                bass_response: 2.0,
                mid_response: 1.2,
                high_response: 0.8,
                spectrum_enabled: false,
                spectrum_style: SpectrumStyle::Bars,
                waveform_enabled: false,
                waveform_style: WaveformStyle::Line,
            },
            fractals: FractalPreset {
                enabled: false,
                ..Default::default()
            },
        }
    }
    
    /// Audio Fractals preset (from fractal_sugar)
    /// Raymarched fractals with audio-reactive coloring
    pub fn preset_audio_fractals() -> Self {
        Self {
            preset_type: PresetType::AudioFractals,
            particles: ParticlePreset {
                count: 1200,
                min_size: 4.0,
                max_size: 12.0,
                speed: 0.3,
                mode: ParticleMode::Orbit,
                shape: ParticleShape::Glow,
                glow_intensity: 0.6,
                damping: 4.0,
                beat_pulse: 0.5,
                attractor_strength: 0.2,
                curl_strength: 0.4,
                spring_mode: true,
                use_3d: true,
            },
            visuals: VisualsPreset {
                bloom_enabled: true,
                bloom_intensity: 0.8,
                bloom_threshold: 0.5,
                bloom_radius: 25.0,
                bloom_mip_levels: 5,
                motion_blur: 0.3,
                vignette: 0.5,
                chromatic_aberration: 0.0,
                echo_enabled: true,
                echo_zoom: 1.005,
                echo_rotation: 0.002,
                kaleidoscope_enabled: true,
                kaleidoscope_segments: 6,
                kaleidoscope_animated: true,
            },
            audio: AudioPreset {
                smoothing: 0.7,
                beat_sensitivity: 0.4,
                beat_attack: 0.5,
                beat_decay: 0.2,
                bass_response: 1.5,
                mid_response: 1.0,
                high_response: 0.8,
                spectrum_enabled: false,
                spectrum_style: SpectrumStyle::Circle,
                waveform_enabled: false,
                waveform_style: WaveformStyle::Circle,
            },
            fractals: FractalPreset {
                enabled: true,
                fractal_type: FractalType::Mandelbulb,
                iterations: 100,
                reactive_bass: [1.0, 0.3, 0.2],
                reactive_mids: [0.2, 1.0, 0.3],
                reactive_high: [0.3, 0.2, 1.0],
                camera_rotation_speed: 0.02,
                orbit_distance: 2.5,
            },
        }
    }
    
    /// VJ Reactive preset (from visualizer2-canon)
    /// FFT-focused with sharp beat detection
    pub fn preset_vj_reactive() -> Self {
        Self {
            preset_type: PresetType::VJReactive,
            particles: ParticlePreset {
                count: 2000,
                min_size: 3.0,
                max_size: 10.0,
                speed: 0.8,
                mode: ParticleMode::Chaos,
                shape: ParticleShape::Circle,
                glow_intensity: 0.7,
                damping: 2.0,
                beat_pulse: 4.0,
                attractor_strength: 0.4,
                curl_strength: 0.5,
                spring_mode: false,
                use_3d: false,
            },
            visuals: VisualsPreset {
                bloom_enabled: true,
                bloom_intensity: 1.0,
                bloom_threshold: 0.4,
                bloom_radius: 30.0,
                bloom_mip_levels: 5,
                motion_blur: 0.0,
                vignette: 0.3,
                chromatic_aberration: 0.3,
                echo_enabled: false,
                echo_zoom: 1.0,
                echo_rotation: 0.0,
                kaleidoscope_enabled: false,
                kaleidoscope_segments: 6,
                kaleidoscope_animated: false,
            },
            audio: AudioPreset {
                smoothing: 0.2,
                beat_sensitivity: 0.8,
                beat_attack: 0.9,
                beat_decay: 0.4,
                bass_response: 2.0,
                mid_response: 1.5,
                high_response: 1.2,
                spectrum_enabled: true,
                spectrum_style: SpectrumStyle::MirrorBars,
                waveform_enabled: true,
                waveform_style: WaveformStyle::Line,
            },
            fractals: FractalPreset {
                enabled: false,
                ..Default::default()
            },
        }
    }
    
    /// Bloom Intensive preset (from WebGPU-Bloom)
    /// Multi-pass bloom with emissive particles
    pub fn preset_bloom_intensive() -> Self {
        Self {
            preset_type: PresetType::BloomIntensive,
            particles: ParticlePreset {
                count: 3000,
                min_size: 3.0,
                max_size: 8.0,
                speed: 0.6,
                mode: ParticleMode::Calm,
                shape: ParticleShape::Glow,
                glow_intensity: 1.0,
                damping: 2.5,
                beat_pulse: 2.0,
                attractor_strength: 0.3,
                curl_strength: 0.5,
                spring_mode: false,
                use_3d: false,
            },
            visuals: VisualsPreset {
                bloom_enabled: true,
                bloom_intensity: 2.0,
                bloom_threshold: 0.3,
                bloom_radius: 50.0,
                bloom_mip_levels: 7,
                motion_blur: 0.15,
                vignette: 0.4,
                chromatic_aberration: 0.1,
                echo_enabled: true,
                echo_zoom: 1.01,
                echo_rotation: 0.005,
                kaleidoscope_enabled: false,
                kaleidoscope_segments: 6,
                kaleidoscope_animated: false,
            },
            audio: AudioPreset {
                smoothing: 0.5,
                beat_sensitivity: 0.5,
                beat_attack: 0.6,
                beat_decay: 0.3,
                bass_response: 1.5,
                mid_response: 1.0,
                high_response: 0.8,
                spectrum_enabled: true,
                spectrum_style: SpectrumStyle::Circle,
                waveform_enabled: false,
                waveform_style: WaveformStyle::Circle,
            },
            fractals: FractalPreset {
                enabled: false,
                ..Default::default()
            },
        }
    }
    
    /// Get preset by type
    pub fn from_type(preset_type: PresetType) -> Self {
        match preset_type {
            PresetType::Default => Self::preset_default(),
            PresetType::ParticlesGPU => Self::preset_particles_gpu(),
            PresetType::AudioFractals => Self::preset_audio_fractals(),
            PresetType::VJReactive => Self::preset_vj_reactive(),
            PresetType::BloomIntensive => Self::preset_bloom_intensive(),
            PresetType::Custom => Self::preset_default(),
        }
    }
}
