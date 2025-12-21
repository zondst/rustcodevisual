//! Configuration System for Particle Studio RS
//! Complete configuration matching Python visualaudio-optimized

use serde::{Deserialize, Serialize};

// ============================================================================
// Enums
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum ParticleMode {
    Chaos,
    Calm,
    Cinematic,
    Orbit,
    DeathSpiral,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum ParticleShape {
    Point,
    Circle,
    Ring,
    Star,
    Diamond,
    Triangle,
    Spark,
    Glow,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum SpectrumStyle {
    Bars,
    MirrorBars,
    Line,
    Circle,
    Waterfall,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum WaveformStyle {
    Line,
    Bars,
    Circle,
    Mirror,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum BlendMode {
    Add,
    Screen,
    Multiply,
    Overlay,
}

// ============================================================================
// Tone Mapping
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum TonemapMethod {
    None,
    Reinhard,
    ACES,
    AgX,
}

// ============================================================================
// Color Scheme
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ColorScheme {
    pub name: String,
    pub particles: Vec<[u8; 3]>,
    pub background: [u8; 3],
    pub nebula_dark: [u8; 3],
    pub nebula_mid: [u8; 3],
    pub nebula_bright: [u8; 3],
    pub waveform: [u8; 3],
    pub spectrum_low: [u8; 3],
    pub spectrum_high: [u8; 3],
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self::cosmic()
    }
}

impl ColorScheme {
    pub fn cosmic() -> Self {
        Self {
            name: "Cosmic".to_string(),
            particles: vec![
                [255, 100, 150],
                [100, 150, 255],
                [150, 255, 200],
                [255, 200, 100],
            ],
            background: [5, 5, 15],
            nebula_dark: [20, 10, 40],
            nebula_mid: [60, 20, 100],
            nebula_bright: [150, 50, 200],
            waveform: [100, 200, 255],
            spectrum_low: [50, 150, 255],
            spectrum_high: [255, 50, 150],
        }
    }

    pub fn fire() -> Self {
        Self {
            name: "Fire".to_string(),
            particles: vec![[255, 200, 50], [255, 100, 0], [255, 50, 0], [200, 0, 0]],
            background: [10, 5, 0],
            nebula_dark: [30, 10, 0],
            nebula_mid: [100, 30, 0],
            nebula_bright: [255, 100, 0],
            waveform: [255, 150, 50],
            spectrum_low: [255, 100, 0],
            spectrum_high: [255, 255, 100],
        }
    }

    pub fn ocean() -> Self {
        Self {
            name: "Ocean".to_string(),
            particles: vec![
                [50, 150, 255],
                [0, 200, 200],
                [100, 255, 200],
                [0, 100, 150],
            ],
            background: [0, 10, 20],
            nebula_dark: [0, 20, 40],
            nebula_mid: [0, 50, 100],
            nebula_bright: [50, 150, 200],
            waveform: [100, 255, 255],
            spectrum_low: [0, 100, 200],
            spectrum_high: [100, 255, 200],
        }
    }

    pub fn aurora() -> Self {
        Self {
            name: "Aurora".to_string(),
            particles: vec![
                [50, 255, 150],
                [100, 200, 255],
                [200, 100, 255],
                [255, 150, 200],
            ],
            background: [5, 10, 15],
            nebula_dark: [10, 30, 20],
            nebula_mid: [30, 100, 80],
            nebula_bright: [100, 255, 150],
            waveform: [150, 255, 200],
            spectrum_low: [50, 200, 150],
            spectrum_high: [200, 100, 255],
        }
    }

    pub fn neon() -> Self {
        Self {
            name: "Neon".to_string(),
            particles: vec![[255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 255, 0]],
            background: [0, 0, 0],
            nebula_dark: [10, 0, 20],
            nebula_mid: [50, 0, 100],
            nebula_bright: [150, 0, 255],
            waveform: [0, 255, 255],
            spectrum_low: [255, 0, 255],
            spectrum_high: [0, 255, 255],
        }
    }

    pub fn all_schemes() -> Vec<ColorScheme> {
        vec![
            Self::cosmic(),
            Self::fire(),
            Self::ocean(),
            Self::aurora(),
            Self::neon(),
        ]
    }
}

// ============================================================================
// Particle Configuration
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ParticleConfig {
    pub enabled: bool,
    pub count: usize,
    pub min_size: f32,
    pub max_size: f32,
    pub speed: f32,
    pub trail_length: usize,
    pub mode: ParticleMode,
    pub shape: ParticleShape,
    pub blend_mode: BlendMode,
    pub spread: f32,
    pub gravity: f32,
    pub size_variation: f32,
    pub breathing_scale: f32,
    pub orbit_speed: f32,
    pub beat_size_pulse: f32,
    // Smoothness controls
    pub glow_intensity: f32,
    pub damping: f32,
    // Audio-reactive spawning
    pub audio_reactive_spawn: bool,
    pub audio_spawn_threshold: f32,
    pub fade_without_audio: bool,
    // Volumetric rendering
    pub volumetric_rendering: bool,
    pub volumetric_steps: u32,
    pub beat_burst_strength: f32, // Velocity burst on beats (0.0-3.0)
    pub fade_attack_speed: f32,   // How fast particles appear (1.0-10.0)
    pub fade_release_speed: f32,  // How fast particles fade (0.5-5.0)
    pub spawn_from_center: bool,  // Spawn near center (burst outward)
    pub spawn_radius: f32,        // Initial spawn radius from center
    // Adaptive audio normalization (NEW)
    pub adaptive_audio_enabled: bool, // Use dynamic normalization
    pub adaptive_window_secs: f32,    // Sliding window duration
    pub bass_sensitivity: f32,        // Bass frequency weight (0.5-2.0)
    pub mid_sensitivity: f32,         // Mid frequency weight
    pub high_sensitivity: f32,        // High frequency weight
    pub adaptive_strength: f32,       // 0=raw audio, 1=balanced, 2=fully normalized

    // New quality & density controls
    #[serde(default)]
    pub hdr_max_brightness: f32, // Max brightness for HDR (2.0-5.0)
    #[serde(default)]
    pub density_limit_enabled: bool, // Enable screen-space density control
    #[serde(default)]
    pub density_cell_size: u32, // Cell size for density check (16-32)
    #[serde(default)]
    pub repulsion_enabled: bool, // Prevent particle clumping
    #[serde(default)]
    pub repulsion_strength: f32, // Repulsion force (0.1-0.5)
    #[serde(default)]
    pub repulsion_radius: f32, // Repulsion radius (20-50)
    #[serde(default)]
    pub depth_sort_enabled: bool, // Sort particles by brightness
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            count: 1500,
            min_size: 2.0,
            max_size: 6.0,
            speed: 0.8,
            trail_length: 8,
            mode: ParticleMode::Chaos,
            shape: ParticleShape::Circle,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.0,
            size_variation: 0.4,
            breathing_scale: 0.2,
            orbit_speed: 0.1,
            beat_size_pulse: 1.5,
            glow_intensity: 0.5,
            damping: 2.0,
            audio_reactive_spawn: true,
            audio_spawn_threshold: 0.02,
            fade_without_audio: true,
            volumetric_rendering: true,
            volumetric_steps: 16,
            // Audio-physics defaults
            beat_burst_strength: 1.5,
            fade_attack_speed: 5.0,
            fade_release_speed: 2.0,
            spawn_from_center: true,
            spawn_radius: 80.0,
            // Adaptive audio defaults
            adaptive_audio_enabled: true, // ON by default
            adaptive_window_secs: 3.0,
            bass_sensitivity: 1.2,
            mid_sensitivity: 1.0,
            high_sensitivity: 0.8,
            adaptive_strength: 0.5, // Balanced by default

            hdr_max_brightness: 5.0,
            density_limit_enabled: true,
            density_cell_size: 16,
            repulsion_enabled: false,
            repulsion_strength: 0.2,
            repulsion_radius: 30.0,
            depth_sort_enabled: true,
        }
    }
}

// ============================================================================
// Attractor Types (from fractal_sugar/rust-particles-main)
// ============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum AttractorType {
    /// Point attractor - pulls particles toward a single point
    Point,
    /// Curl attractor - creates swirling motion (cross product force)
    Curl,
    /// Big boomer - inverse-power repulsion on beats
    BigBoomer,
    /// Sphere collider - particles bounce off spheres
    Collider,
}

impl Default for AttractorType {
    fn default() -> Self {
        Self::Point
    }
}

// ============================================================================
// Physics Configuration
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct PhysicsConfig {
    pub curl_noise_scale: f32,
    pub curl_noise_strength: f32,
    pub turbulence_strength: f32,
    pub attractor_strength: f32,
    pub attractor_count: usize,
    pub attractor_type: AttractorType,
    pub damping: f32,
    pub bounds_mode: String,
    pub wind_x: f32,
    pub wind_y: f32,
    /// Spring mode - particles spring back to fixed positions (jello effect)
    pub spring_mode: bool,
    pub spring_coefficient: f32,
    /// 3D mode - use cube positions instead of square
    pub use_3d: bool,
    /// Big boomer strength on beats
    pub big_boomer_strength: f32,
    /// Curl attractors for rotational forces
    pub curl_attractor_count: usize,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            curl_noise_scale: 3.0,
            curl_noise_strength: 0.6,
            turbulence_strength: 0.4,
            attractor_strength: 0.3,
            attractor_count: 1,
            attractor_type: AttractorType::Point,
            damping: 0.97,
            bounds_mode: "wrap".to_string(),
            wind_x: 0.0,
            wind_y: 0.0,
            spring_mode: false,
            spring_coefficient: 0.5,
            use_3d: false,
            big_boomer_strength: 2.4,
            curl_attractor_count: 2,
        }
    }
}

// ============================================================================
// Death Spiral Configuration (Ant Mill / Death Dance effect)
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct DeathSpiralConfig {
    /// Number of concentric spiral rings (2-8)
    pub ring_count: usize,

    /// Base radius of the innermost ring
    pub inner_radius: f32,

    /// Multiplier for distance between rings (1.2-2.0)
    pub ring_spacing: f32,

    /// Base rotation speed in radians/sec
    pub base_rotation_speed: f32,

    /// Alternate rotation direction for each ring
    pub alternate_direction: bool,

    /// Audio influence on rotation speed (0.0-1.0)
    pub audio_speed_influence: f32,

    /// Audio influence on ring radius (0.0-1.0)
    pub audio_radius_influence: f32,

    /// Trail intensity for pheromone effect (0.0-1.0)
    pub trail_intensity: f32,

    /// Particles per ring
    pub particles_per_ring: usize,

    /// Wave amplitude for organic breathing effect (0.0-0.5)
    pub wave_amplitude: f32,

    /// Wave frequency (number of waves around the ring)
    pub wave_frequency: f32,

    /// Spiral tightness - adds inward/outward spiral motion (0.0-0.3)
    pub spiral_tightness: f32,

    /// Size pulse strength on beat (0.0-1.0)
    pub beat_pulse_strength: f32,

    /// Follow strength - how strongly particles follow target position (0.1-0.6)
    pub follow_strength: f32,
}

impl Default for DeathSpiralConfig {
    fn default() -> Self {
        Self {
            ring_count: 5,
            inner_radius: 100.0,
            ring_spacing: 1.5,
            base_rotation_speed: 0.8,
            alternate_direction: true,
            audio_speed_influence: 0.6,
            audio_radius_influence: 0.3,
            trail_intensity: 0.7,
            particles_per_ring: 60,
            wave_amplitude: 0.15,
            wave_frequency: 3.0,
            spiral_tightness: 0.1,
            beat_pulse_strength: 0.4,
            follow_strength: 0.3,
        }
    }
}

// ============================================================================
// Trail System Configuration (Pheromone trails)
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct TrailConfig {
    /// Enable trail rendering
    pub enabled: bool,

    /// Maximum trail points per particle
    pub max_length: usize,

    /// Trail fade speed (how fast trails disappear)
    pub fade_speed: f32,

    /// Trail spawn rate (points per second)
    pub spawn_rate: f32,

    /// Trail width relative to particle size
    pub width_scale: f32,

    /// Glow effect on trails
    pub glow_enabled: bool,

    /// Trail opacity multiplier
    pub opacity: f32,
}

impl Default for TrailConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_length: 15,
            fade_speed: 2.0,
            spawn_rate: 30.0,
            width_scale: 0.6,
            glow_enabled: true,
            opacity: 0.7,
        }
    }
}

// ============================================================================
// Connection System Configuration (Energy links between particles)
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ConnectionConfig {
    /// Enable particle connections
    pub enabled: bool,

    /// Maximum distance for connection
    pub max_distance: f32,

    /// Maximum connections per particle
    pub max_connections: usize,

    /// Connection line opacity
    pub opacity: f32,

    /// Connection line thickness
    pub thickness: f32,

    /// React to audio (connections pulse with music)
    pub audio_reactive: bool,

    /// Enable gradient between particle colors
    pub gradient_enabled: bool,

    // New optimized connection params
    pub density_limit: f32,      // Max connections density per pixel area
    pub min_particle_alpha: f32, // Don't connect faded particles
    pub curve_enabled: bool,     // Curved lines
    pub fade_by_distance: bool,  // Fade opacity based on distance
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_distance: 80.0,
            max_connections: 3,
            opacity: 0.3,
            thickness: 1.0,
            audio_reactive: true,
            gradient_enabled: true,
            density_limit: 5.0,
            min_particle_alpha: 0.1,
            curve_enabled: false,
            fade_by_distance: true,
        }
    }
}

// ============================================================================
// Cinematic Effects Configuration
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct CinematicConfig {
    /// Enable chromatic aberration
    pub chromatic_enabled: bool,

    /// Chromatic aberration intensity (0.0-1.0)
    pub chromatic_intensity: f32,

    /// Radial chromatic (stronger towards edges)
    pub chromatic_radial: bool,

    /// Enable film grain
    pub grain_enabled: bool,

    /// Film grain intensity (0.0-1.0)
    pub grain_intensity: f32,

    /// Film grain size (1.0-3.0)
    pub grain_size: f32,

    /// Enable vignette
    pub vignette_enabled: bool,

    /// Vignette intensity (0.0-1.0)
    pub vignette_intensity: f32,

    /// Vignette softness (0.0-1.0)
    pub vignette_softness: f32,

    /// Enable letterbox (cinematic bars)
    pub letterbox_enabled: bool,

    /// Letterbox aspect ratio (2.35 for anamorphic, 1.85 for standard)
    pub letterbox_ratio: f32,

    /// Color temperature (-1.0 cold to 1.0 warm)
    pub color_temperature: f32,
}

impl Default for CinematicConfig {
    fn default() -> Self {
        Self {
            chromatic_enabled: false,
            chromatic_intensity: 0.3,
            chromatic_radial: true,
            grain_enabled: false,
            grain_intensity: 0.1,
            grain_size: 1.5,
            vignette_enabled: true,
            vignette_intensity: 0.3,
            vignette_softness: 0.5,
            letterbox_enabled: false,
            letterbox_ratio: 2.35,
            color_temperature: 0.0,
        }
    }
}

// ============================================================================
// Audio Configuration
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct AudioConfig {
    pub smoothing: f32,
    pub beat_sensitivity: f32,
    pub bass_response: f32,
    pub mid_response: f32,
    pub high_response: f32,
    pub beat_explosion_strength: f32,
    pub beat_size_pulse: f32,
    pub frequency_bands: usize,
    // NEW: Beat attack/decay for smooth transitions
    pub beat_attack: f32, // How fast beat rises (0.3-1.0)
    pub beat_decay: f32,  // How slow beat falls (0.1-0.5)
    #[serde(default)]
    pub latency_ms: f32, // Audio latency compensation in milliseconds
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            smoothing: 0.5, // Increased for smoother default
            beat_sensitivity: 0.5,
            bass_response: 1.5,
            mid_response: 1.0,
            high_response: 0.8,
            beat_explosion_strength: 4.0,
            beat_size_pulse: 2.0,
            frequency_bands: 64,
            beat_attack: 0.6, // Moderate attack
            beat_decay: 0.25, // Slow decay
            latency_ms: 0.0,
        }
    }
}

// ============================================================================
// Visual / Post-Processing Configuration
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VisualConfig {
    // Bloom
    pub bloom_enabled: bool,
    pub bloom_intensity: f32,
    pub bloom_radius: f32,
    pub bloom_threshold: f32,
    /// Multi-pass bloom with mipchain (from WebGPU-Bloom)
    pub bloom_mip_levels: u32,
    /// Bloom knee for soft threshold transition
    pub bloom_knee: f32,

    // Glow
    pub glow_intensity: f32,
    pub glow_radius: f32,

    // Effects
    pub motion_blur: f32,
    pub vignette_strength: f32,
    pub film_grain: f32,
    pub chromatic_aberration: f32,
    pub scanlines: bool,
    pub scanline_intensity: f32,

    // Tone mapping
    pub exposure: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub gamma: f32,
    pub tonemap_method: TonemapMethod,

    // MilkDrop effects
    pub echo_enabled: bool,
    pub echo_zoom: f32,
    pub echo_rotation: f32,
    pub echo_alpha: f32,

    pub kaleidoscope_enabled: bool,
    pub kaleidoscope_segments: usize,
    /// Animated kaleidoscope (from fractal_sugar)
    pub kaleidoscope_animated: bool,
    pub kaleidoscope_speed: f32,

    pub radial_blur_enabled: bool,
    pub radial_blur_amount: f32,

    pub color_shift_enabled: bool,
    pub feedback_enabled: bool,

    // Fractal background (from fractal_sugar)
    pub fractal_enabled: bool,
    pub fractal_type: usize, // 0-6 matching FractalType enum
    pub fractal_intensity: f32,

    // UI elements
    pub show_audio_meters: bool, // Show Bass/Mid/High bars at bottom-left
}

impl Default for VisualConfig {
    fn default() -> Self {
        Self {
            bloom_enabled: true,
            bloom_intensity: 0.6,
            bloom_radius: 25.0,
            bloom_threshold: 0.5,
            bloom_mip_levels: 5,
            bloom_knee: 0.2,

            glow_intensity: 1.2,
            glow_radius: 15.0,

            motion_blur: 0.2,
            vignette_strength: 0.3,
            film_grain: 0.02,
            chromatic_aberration: 0.0,
            scanlines: false,
            scanline_intensity: 0.1,

            exposure: 1.0,
            contrast: 1.0,
            saturation: 1.0,
            gamma: 1.0,
            tonemap_method: TonemapMethod::ACES,

            echo_enabled: false,
            echo_zoom: 1.02,
            echo_rotation: 0.01,
            echo_alpha: 0.5,

            kaleidoscope_enabled: false,
            kaleidoscope_segments: 6,
            kaleidoscope_animated: false,
            kaleidoscope_speed: 0.275,

            radial_blur_enabled: false,
            radial_blur_amount: 0.0,

            color_shift_enabled: false,
            feedback_enabled: false,

            fractal_enabled: false,
            fractal_type: 0,
            fractal_intensity: 0.5,

            show_audio_meters: true, // Show by default
        }
    }
}

// ============================================================================
// Waveform Configuration
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct WaveformConfig {
    pub enabled: bool,
    pub style: WaveformStyle,
    pub thickness: f32,
    pub amplitude: f32,
    pub position_x: f32,
    pub position_y: f32,
    pub smoothing: usize,
    pub mirror: bool,
    pub circular_radius: f32,
}

impl Default for WaveformConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            style: WaveformStyle::Line,
            thickness: 3.0,
            amplitude: 200.0,
            position_x: 0.5,
            position_y: 0.5,
            smoothing: 5,
            mirror: true,
            circular_radius: 0.3,
        }
    }
}

// ============================================================================
// Spectrum Configuration
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SpectrumConfig {
    pub enabled: bool,
    pub style: SpectrumStyle,
    pub bar_count: usize,
    pub bar_width: f32,
    pub bar_spacing: f32,
    pub bar_height_scale: f32,
    pub smoothing: f32,
    pub logarithmic: bool,
    pub mirror: bool,
    pub position_x: f32,
    pub position_y: f32,
}

impl Default for SpectrumConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            style: SpectrumStyle::Bars,
            bar_count: 64,
            bar_width: 0.8,
            bar_spacing: 0.2,
            bar_height_scale: 1.0,
            smoothing: 0.5,
            logarithmic: true,
            mirror: true,
            position_x: 0.5,
            position_y: 0.0,
        }
    }
}

// ============================================================================
// Export Configuration
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ExportConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub codec: String,
    pub preset: String,
    pub crf: u32,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fps: 30,
            codec: "libx264".to_string(),
            preset: "medium".to_string(),
            crf: 18,
        }
    }
}

// ============================================================================
// Background Configuration
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct BackgroundConfig {
    pub animated: bool,
    pub intensity: f32,
}

impl Default for BackgroundConfig {
    fn default() -> Self {
        Self {
            animated: true,
            intensity: 0.3,
        }
    }
}

// ============================================================================
// Main App Configuration
// ============================================================================

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
pub struct AppConfig {
    pub particles: ParticleConfig,
    pub physics: PhysicsConfig,
    pub audio: AudioConfig,
    pub visual: VisualConfig,
    pub waveform: WaveformConfig,
    pub spectrum: SpectrumConfig,
    pub export: ExportConfig,
    pub background: BackgroundConfig,
    pub color_scheme_index: usize,
    // New configurations
    #[serde(default)]
    pub death_spiral: DeathSpiralConfig,
    #[serde(default)]
    pub trails: TrailConfig,
    #[serde(default)]
    pub connections: ConnectionConfig,
    #[serde(default)]
    pub cinematic: CinematicConfig,
}

impl AppConfig {
    pub fn get_color_scheme(&self) -> ColorScheme {
        let schemes = ColorScheme::all_schemes();
        schemes
            .get(self.color_scheme_index)
            .cloned()
            .unwrap_or_default()
    }

    #[allow(dead_code)]
    pub fn set_color_scheme(&mut self, index: usize) {
        let schemes = ColorScheme::all_schemes();
        if index < schemes.len() {
            self.color_scheme_index = index;
        }
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let config = serde_json::from_str(&json)?;
        Ok(config)
    }

    /// Get all available preset names
    pub fn preset_names() -> Vec<&'static str> {
        vec![
            "Default",
            "Cinematic",
            "Ambient",
            "Energetic",
            "Minimal",
            "Psychedelic",
            "Starfield",
            "Fractal Flow",
            "GPU Sim",
            "Audio Reactive",
            "Spectrum Bars",
            // New sample-inspired presets
            "Particles GPU",
            "Audio Fractals",
            "VJ Reactive",
            "Bloom Intensive",
            // Death Spiral presets
            "Ant Dance",
            "Hypnotic Vortex",
        ]
    }

    /// Apply a preset by name
    pub fn apply_preset(&mut self, name: &str) {
        match name {
            "Default" => self.preset_default(),
            "Cinematic" => self.preset_cinematic(),
            "Ambient" => self.preset_ambient(),
            "Energetic" => self.preset_energetic(),
            "Minimal" => self.preset_minimal(),
            "Psychedelic" => self.preset_psychedelic(),
            "Starfield" => self.preset_starfield(),
            "Fractal Flow" => self.preset_fractal_flow(),
            "GPU Sim" => self.preset_gpu_sim(),
            "Audio Reactive" => self.preset_audio_reactive(),
            "Spectrum Bars" => self.preset_spectrum_bars(),
            "Particles GPU" => self.preset_particles_gpu(),
            "Audio Fractals" => self.preset_audio_fractals(),
            "VJ Reactive" => self.preset_vj_reactive(),
            "Bloom Intensive" => self.preset_bloom_intensive(),
            "Ant Dance" => self.preset_ant_dance(),
            "Hypnotic Vortex" => self.preset_hypnotic_vortex(),
            _ => {}
        }
    }

    pub fn preset_default(&mut self) {
        self.particles = ParticleConfig::default();
        self.visual = VisualConfig::default();
        self.audio = AudioConfig::default();
        self.spectrum = SpectrumConfig::default();
        self.waveform = WaveformConfig::default();
        self.connections = ConnectionConfig::default();
        self.trails = TrailConfig::default();
        self.death_spiral = DeathSpiralConfig::default();
        self.color_scheme_index = 0; // Cosmic
    }

    /// Cinematic preset - slow, elegant particles with smooth motion
    pub fn preset_cinematic(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 200,    // Much fewer particles
            min_size: 2.0, // Smaller
            max_size: 6.0, // Much smaller (was 15!)
            speed: 0.05,   // Very slow (was 0.2)
            trail_length: 8,
            mode: ParticleMode::Cinematic,
            shape: ParticleShape::Glow,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.0,
            size_variation: 0.3,
            breathing_scale: 0.15, // Very subtle breathing
            orbit_speed: 0.02,
            beat_size_pulse: 0.3, // Very subtle beat response (was 1.0)
            glow_intensity: 0.4,  // Reduced glow
            damping: 4.0,         // High damping for smooth stops
            ..Default::default()
        };

        self.visual = VisualConfig {
            bloom_enabled: true,
            bloom_intensity: 0.8, // Reduced (was 1.2)
            bloom_radius: 25.0,   // Smaller (was 35)
            bloom_threshold: 0.5, // Higher threshold
            glow_intensity: 1.0,
            glow_radius: 15.0,
            motion_blur: 0.4,
            vignette_strength: 0.5,
            film_grain: 0.015,
            chromatic_aberration: 0.0,
            scanlines: false,
            scanline_intensity: 0.0,
            exposure: 1.0,
            contrast: 1.0,
            saturation: 0.9,
            gamma: 1.0,
            echo_enabled: true,
            echo_zoom: 1.005,     // Very subtle zoom
            echo_rotation: 0.001, // Very subtle rotation
            echo_alpha: 0.2,
            kaleidoscope_enabled: false,
            kaleidoscope_segments: 6,
            radial_blur_enabled: false,
            radial_blur_amount: 0.0,
            color_shift_enabled: false,
            feedback_enabled: true,
            ..Default::default()
        };

        self.audio.smoothing = 0.85; // Very high smoothing
        self.audio.beat_attack = 0.4; // Slow attack
        self.audio.beat_decay = 0.15; // Very slow decay
        self.audio.beat_size_pulse = 0.3;
        self.spectrum.enabled = false;
        self.waveform.enabled = true;
        self.waveform.style = WaveformStyle::Line;
        self.waveform.amplitude = 80.0;
        self.color_scheme_index = 0; // Cosmic
    }

    /// Ambient preset - calm, floating particles
    pub fn preset_ambient(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 800,
            min_size: 2.0,
            max_size: 8.0,
            speed: 0.15,
            trail_length: 6,
            mode: ParticleMode::Calm,
            shape: ParticleShape::Circle,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: -0.02,
            size_variation: 0.4,
            breathing_scale: 0.8,
            orbit_speed: 0.02,
            beat_size_pulse: 0.3,
            glow_intensity: 0.5,
            damping: 3.0,
            ..Default::default()
        };

        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 0.8;
        self.visual.motion_blur = 0.4;
        self.visual.vignette_strength = 0.4;
        self.visual.echo_enabled = false;
        self.audio.smoothing = 0.8;
        self.audio.beat_attack = 0.5;
        self.audio.beat_decay = 0.2;
        self.spectrum.enabled = false;
        self.waveform.enabled = true;
        self.waveform.style = WaveformStyle::Circle;
        self.color_scheme_index = 3; // Aurora
    }

    /// Energetic preset - fast, reactive particles
    pub fn preset_energetic(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 3000,
            min_size: 1.5,
            max_size: 6.0,
            speed: 2.0,
            trail_length: 4,
            mode: ParticleMode::Chaos,
            shape: ParticleShape::Spark,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.0,
            size_variation: 0.5,
            breathing_scale: 0.2,
            orbit_speed: 0.3,
            beat_size_pulse: 3.0,
            glow_intensity: 0.8,
            damping: 1.5, // Low damping for fast motion
            ..Default::default()
        };

        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 1.0;
        self.visual.motion_blur = 0.15;
        self.visual.chromatic_aberration = 0.3;
        self.visual.echo_enabled = false;
        self.audio.smoothing = 0.2;
        self.audio.beat_attack = 0.8;
        self.audio.beat_decay = 0.4;
        self.audio.beat_explosion_strength = 6.0;
        self.spectrum.enabled = true;
        self.spectrum.style = SpectrumStyle::MirrorBars;
        self.waveform.enabled = false;
        self.color_scheme_index = 1; // Fire
    }

    /// Minimal preset - clean, simple visualization
    pub fn preset_minimal(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 300,
            min_size: 2.0,
            max_size: 4.0,
            speed: 0.3,
            trail_length: 2,
            mode: ParticleMode::Calm,
            shape: ParticleShape::Point,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.0,
            size_variation: 0.2,
            breathing_scale: 0.1,
            orbit_speed: 0.1,
            beat_size_pulse: 0.5,
            glow_intensity: 0.3,
            damping: 2.5,
            ..Default::default()
        };

        self.visual = VisualConfig {
            bloom_enabled: false,
            bloom_intensity: 0.0,
            bloom_radius: 10.0,
            bloom_threshold: 0.8,
            glow_intensity: 0.5,
            glow_radius: 5.0,
            motion_blur: 0.0,
            vignette_strength: 0.2,
            film_grain: 0.0,
            chromatic_aberration: 0.0,
            scanlines: false,
            scanline_intensity: 0.0,
            exposure: 1.0,
            contrast: 1.0,
            saturation: 1.0,
            gamma: 1.0,
            echo_enabled: false,
            echo_zoom: 1.0,
            echo_rotation: 0.0,
            echo_alpha: 0.0,
            kaleidoscope_enabled: false,
            kaleidoscope_segments: 6,
            radial_blur_enabled: false,
            radial_blur_amount: 0.0,
            color_shift_enabled: false,
            feedback_enabled: false,
            ..Default::default()
        };

        self.audio.smoothing = 0.6;
        self.audio.beat_attack = 0.6;
        self.audio.beat_decay = 0.3;
        self.spectrum.enabled = true;
        self.spectrum.style = SpectrumStyle::Line;
        self.waveform.enabled = true;
        self.waveform.style = WaveformStyle::Line;
        self.color_scheme_index = 2; // Ocean
    }

    /// Psychedelic preset - colorful, trippy effects
    pub fn preset_psychedelic(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 1500,
            min_size: 3.0,
            max_size: 12.0,
            speed: 0.8,
            trail_length: 10,
            mode: ParticleMode::Orbit,
            shape: ParticleShape::Ring,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.0,
            size_variation: 0.6,
            breathing_scale: 0.6,
            orbit_speed: 0.15,
            beat_size_pulse: 2.0,
            glow_intensity: 0.7,
            damping: 2.0,
            ..Default::default()
        };

        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 1.5;
        self.visual.chromatic_aberration = 0.5;
        self.visual.echo_enabled = true;
        self.visual.echo_zoom = 1.02;
        self.visual.echo_rotation = 0.02;
        self.visual.echo_alpha = 0.4;
        self.visual.kaleidoscope_enabled = true;
        self.visual.kaleidoscope_segments = 6;
        self.visual.color_shift_enabled = true;
        self.audio.smoothing = 0.4;
        self.audio.beat_attack = 0.7;
        self.audio.beat_decay = 0.3;
        self.spectrum.enabled = true;
        self.spectrum.style = SpectrumStyle::Circle;
        self.waveform.enabled = false;
        self.color_scheme_index = 4; // Neon
    }

    /// Starfield preset - space-like floating stars
    pub fn preset_starfield(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 1000,
            min_size: 1.0,
            max_size: 4.0,
            speed: 0.1,
            trail_length: 0,
            mode: ParticleMode::Calm,
            shape: ParticleShape::Star,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.0,
            size_variation: 0.7,
            breathing_scale: 1.0,
            orbit_speed: 0.01,
            beat_size_pulse: 1.5,
            glow_intensity: 0.5,
            damping: 3.0,
            ..Default::default()
        };

        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 0.6;
        self.visual.bloom_threshold = 0.4;
        self.visual.motion_blur = 0.0;
        self.visual.vignette_strength = 0.6;
        self.visual.echo_enabled = false;
        self.audio.smoothing = 0.6;
        self.audio.beat_attack = 0.5;
        self.audio.beat_decay = 0.2;
        self.spectrum.enabled = false;
        self.waveform.enabled = true;
        self.waveform.style = WaveformStyle::Line;
        self.waveform.amplitude = 80.0;
        self.color_scheme_index = 0; // Cosmic
    }

    /// Fractal Flow preset - inspired by fractal_sugar
    /// Orbital particles with echo/feedback for trippy trails
    pub fn preset_fractal_flow(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 1200,
            min_size: 4.0,
            max_size: 12.0,
            speed: 0.3,
            trail_length: 15,
            mode: ParticleMode::Orbit,
            shape: ParticleShape::Glow,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.0,
            size_variation: 0.5,
            breathing_scale: 0.4,
            orbit_speed: 0.08,
            beat_size_pulse: 2.5,
            glow_intensity: 1.0,
            damping: 2.5,
            ..Default::default()
        };

        // Explicitly disable connections
        self.connections.enabled = false;

        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 1.3;
        self.visual.motion_blur = 0.3;
        self.visual.echo_enabled = true;
        self.visual.echo_zoom = 1.008;
        self.visual.echo_rotation = 0.01;
        self.visual.echo_alpha = 0.35;
        self.visual.feedback_enabled = true;
        self.visual.vignette_strength = 0.4;
        self.audio.smoothing = 0.5;
        self.audio.beat_explosion_strength = 4.0;
        self.spectrum.enabled = false;
        self.waveform.enabled = false;
        self.color_scheme_index = 0; // Cosmic
    }

    /// GPU Sim preset - inspired by rust-particles-main
    /// Many particles with attractor physics and heavy bloom
    pub fn preset_gpu_sim(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 5000,
            min_size: 2.0,
            max_size: 6.0,
            speed: 1.5,
            trail_length: 3,
            mode: ParticleMode::Chaos,
            shape: ParticleShape::Glow,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.02,
            size_variation: 0.4,
            breathing_scale: 0.2,
            orbit_speed: 0.2,
            beat_size_pulse: 2.0,
            glow_intensity: 0.8,
            damping: 1.5,
            ..Default::default()
        };

        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 1.8;
        self.visual.bloom_radius = 45.0;
        self.visual.bloom_threshold = 0.2;
        self.visual.motion_blur = 0.2;
        self.visual.echo_enabled = false;
        self.visual.vignette_strength = 0.3;
        self.audio.smoothing = 0.3;
        self.audio.beat_explosion_strength = 5.0;
        self.spectrum.enabled = false;
        self.waveform.enabled = false;
        self.color_scheme_index = 1; // Fire
    }

    /// Audio Reactive preset - inspired by vis-core beat detection
    /// Highly responsive to music with sharp beat reactions
    pub fn preset_audio_reactive(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 2000,
            min_size: 3.0,
            max_size: 10.0,
            speed: 0.8,
            trail_length: 5,
            mode: ParticleMode::Chaos,
            shape: ParticleShape::Circle,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.0,
            size_variation: 0.5,
            breathing_scale: 0.3,
            orbit_speed: 0.1,
            beat_size_pulse: 4.0, // Strong beat reaction
            glow_intensity: 0.7,
            damping: 2.0,
            ..Default::default()
        };

        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 1.0;
        self.visual.motion_blur = 0.0; // Sharp, no blur
        self.visual.echo_enabled = false;
        self.visual.vignette_strength = 0.3;
        self.audio.smoothing = 0.2; // Low smoothing = more reactive
        self.audio.beat_explosion_strength = 6.0;
        self.spectrum.enabled = true;
        self.spectrum.style = SpectrumStyle::Bars;
        self.waveform.enabled = true;
        self.waveform.style = WaveformStyle::Line;
        self.color_scheme_index = 4; // Neon
    }

    /// Spectrum Bars preset - inspired by visualiser-main
    /// Focus on spectrum visualization with log scaling
    pub fn preset_spectrum_bars(&mut self) {
        self.particles = ParticleConfig {
            enabled: false,
            count: 500,
            min_size: 2.0,
            max_size: 5.0,
            speed: 0.2,
            trail_length: 0,
            mode: ParticleMode::Calm,
            shape: ParticleShape::Circle,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.0,
            size_variation: 0.3,
            breathing_scale: 0.1,
            orbit_speed: 0.05,
            beat_size_pulse: 1.0,
            glow_intensity: 0.5,
            damping: 2.5,
            ..Default::default()
        };

        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 0.8;
        self.visual.motion_blur = 0.1;
        self.visual.echo_enabled = false;
        self.visual.vignette_strength = 0.4;
        self.audio.smoothing = 0.4;
        self.spectrum.enabled = true;
        self.spectrum.style = SpectrumStyle::MirrorBars;
        self.spectrum.bar_count = 64;
        self.spectrum.bar_height_scale = 1.5;
        self.waveform.enabled = true;
        self.waveform.style = WaveformStyle::Mirror;
        self.waveform.amplitude = 100.0;
        self.color_scheme_index = 2; // Ocean
    }

    // ========================================================================
    // NEW SAMPLE-INSPIRED PRESETS
    // ========================================================================

    /// Particles GPU preset - inspired by rust-particles-main
    /// Heavy particle simulation with attractors and intense bloom
    pub fn preset_particles_gpu(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 5000,
            min_size: 2.0,
            max_size: 6.0,
            speed: 1.5,
            trail_length: 3,
            mode: ParticleMode::Chaos,
            shape: ParticleShape::Glow,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.02,
            size_variation: 0.4,
            breathing_scale: 0.2,
            orbit_speed: 0.2,
            beat_size_pulse: 2.5,
            glow_intensity: 0.8,
            damping: 1.5,
            ..Default::default()
        };

        self.physics.attractor_type = AttractorType::Curl;
        self.physics.attractor_strength = 0.5;
        self.physics.curl_attractor_count = 4;
        self.physics.big_boomer_strength = 3.0;

        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 1.8;
        self.visual.bloom_radius = 45.0;
        self.visual.bloom_threshold = 0.2;
        self.visual.bloom_mip_levels = 7;
        self.visual.motion_blur = 0.2;
        self.visual.echo_enabled = false;
        self.visual.vignette_strength = 0.3;
        self.visual.fractal_enabled = false;

        self.audio.smoothing = 0.3;
        self.audio.beat_attack = 0.8;
        self.audio.beat_decay = 0.35;
        self.audio.beat_explosion_strength = 5.0;

        self.spectrum.enabled = false;
        self.waveform.enabled = false;
        self.color_scheme_index = 1; // Fire
    }

    /// Audio Fractals preset - inspired by fractal_sugar
    /// Raymarched fractals with audio-reactive coloring and particles
    pub fn preset_audio_fractals(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 600, // Reduced from 1200
            min_size: 4.0,
            max_size: 12.0,
            speed: 0.3,
            trail_length: 15,
            mode: ParticleMode::Orbit,
            shape: ParticleShape::Glow,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.0,
            size_variation: 0.5,
            breathing_scale: 0.4,
            orbit_speed: 0.08,
            beat_size_pulse: 0.5,
            glow_intensity: 0.8,
            damping: 1.0,
            // Fixes
            spawn_from_center: false,
            spawn_radius: 200.0,
            repulsion_enabled: true,
            repulsion_strength: 0.3,
            repulsion_radius: 30.0,
            ..Default::default()
        };

        // Disable connections (white blob fix)
        self.connections.enabled = false;

        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 0.7;
        self.visual.bloom_radius = 30.0;
        self.visual.bloom_threshold = 0.3;
        self.visual.motion_blur = 0.3;
        self.visual.vignette_strength = 0.5;
        self.visual.echo_enabled = true;
        self.visual.echo_zoom = 1.02;
        self.visual.echo_rotation = 0.002;
        self.visual.echo_alpha = 0.25;
        self.visual.kaleidoscope_enabled = true;
        self.visual.kaleidoscope_segments = 6;
        self.visual.kaleidoscope_animated = true;
        self.visual.kaleidoscope_speed = 0.275;
        self.visual.fractal_enabled = true;
        self.visual.fractal_type = 2; // Mandelbulb
        self.visual.fractal_intensity = 0.6;

        self.audio.smoothing = 0.7;
        self.audio.beat_attack = 0.5;
        self.audio.beat_decay = 0.2;

        self.spectrum.enabled = false;
        self.waveform.enabled = false;
        self.color_scheme_index = 0; // Cosmic
    }

    /// VJ Reactive preset - inspired by visualizer2-canon
    /// Sharp beat detection with FFT spectrum display
    pub fn preset_vj_reactive(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 800, // Reduced from 2000
            min_size: 2.0,
            max_size: 6.0,
            speed: 2.5,
            trail_length: 3,
            mode: ParticleMode::Chaos,
            shape: ParticleShape::Spark,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.0,
            size_variation: 0.8,
            breathing_scale: 0.5,
            orbit_speed: 0.5,
            beat_size_pulse: 3.0,
            glow_intensity: 0.5, // Reduced
            damping: 0.5,
            // New settings
            hdr_max_brightness: 4.0,
            density_limit_enabled: true,
            density_cell_size: 20,
            repulsion_enabled: true,
            repulsion_strength: 0.2,
            repulsion_radius: 20.0,
            ..Default::default()
        };

        // CRITICAL FIX: Disable connections and trails to prevent chaos
        self.connections = ConnectionConfig {
            enabled: false,
            ..Default::default()
        };
        self.trails = TrailConfig {
            enabled: false,
            ..Default::default()
        };

        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 1.0;
        self.visual.bloom_radius = 40.0;
        self.visual.bloom_threshold = 0.4;
        self.visual.motion_blur = 0.5;
        self.visual.vignette_strength = 0.2;
        self.visual.chromatic_aberration = 0.5;
        self.visual.echo_enabled = true;
        self.visual.echo_zoom = 1.15;
        self.visual.echo_alpha = 0.4;
        self.visual.kaleidoscope_enabled = true;
        self.visual.kaleidoscope_segments = 4;

        self.audio.smoothing = 0.2;
        self.audio.beat_explosion_strength = 8.0;
        self.spectrum.enabled = true;
        self.spectrum.style = SpectrumStyle::MirrorBars;
        self.waveform.enabled = true;
        self.waveform.style = WaveformStyle::Circle;
        self.color_scheme_index = 4; // Neon
    }

    /// Bloom Intensive preset - inspired by WebGPU-Bloom
    /// Multi-pass bloom with emissive particles
    pub fn preset_bloom_intensive(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 3000,
            min_size: 3.0,
            max_size: 8.0,
            speed: 0.6,
            trail_length: 8,
            mode: ParticleMode::Calm,
            shape: ParticleShape::Glow,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: -0.01,
            size_variation: 0.4,
            breathing_scale: 0.5,
            orbit_speed: 0.05,
            beat_size_pulse: 2.0,
            glow_intensity: 1.0,
            damping: 2.5,
            ..Default::default()
        };

        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 2.0;
        self.visual.bloom_radius = 50.0;
        self.visual.bloom_threshold = 0.3;
        self.visual.bloom_mip_levels = 7;
        self.visual.bloom_knee = 0.25;
        self.visual.motion_blur = 0.15;
        self.visual.vignette_strength = 0.4;
        self.visual.chromatic_aberration = 0.1;
        self.visual.echo_enabled = true;
        self.visual.echo_zoom = 1.01;
        self.visual.echo_rotation = 0.005;
        self.visual.echo_alpha = 0.3;
        self.visual.fractal_enabled = false;

        self.audio.smoothing = 0.5;
        self.audio.beat_attack = 0.6;
        self.audio.beat_decay = 0.3;

        self.spectrum.enabled = true;
        self.spectrum.style = SpectrumStyle::Circle;
        self.waveform.enabled = false;
        self.color_scheme_index = 3; // Aurora
    }

    // ========================================================================
    // DEATH SPIRAL PRESETS
    // ========================================================================

    /// Ant Dance preset - death spiral behavior
    pub fn preset_ant_dance(&mut self) {
        self.particles = ParticleConfig {
            enabled: true,
            count: 1000, // Reduced from 2000
            min_size: 2.0,
            max_size: 5.0,
            speed: 1.0,
            trail_length: 20,
            mode: ParticleMode::DeathSpiral,
            shape: ParticleShape::Circle,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.0,
            size_variation: 0.3,
            breathing_scale: 0.2,
            orbit_speed: 0.5,
            beat_size_pulse: 1.0,
            glow_intensity: 0.6,
            damping: 0.95,
            ..Default::default()
        };

        // Spiral settings optimized
        self.death_spiral = DeathSpiralConfig {
            ring_count: 5,
            inner_radius: 120.0,
            ring_spacing: 1.8, // Wider spacing
            base_rotation_speed: 0.6,
            alternate_direction: true,
            audio_speed_influence: 0.8,
            audio_radius_influence: 0.4,
            trail_intensity: 0.8,
            particles_per_ring: 40, // Reduced density
            wave_amplitude: 0.2,
            wave_frequency: 3.0,
            spiral_tightness: 0.15,
            beat_pulse_strength: 0.5,
            follow_strength: 0.4,
        };

        // Limited connections
        self.connections = ConnectionConfig {
            enabled: true,
            max_distance: 25.0, // Short range
            max_connections: 1, // Single connection
            opacity: 0.15,      // Low opacity
            thickness: 0.8,
            density_limit: 3.0, // Strict density limit
            min_particle_alpha: 0.2,
            curve_enabled: true,
            fade_by_distance: true,
            ..Default::default()
        };

        // Visual effects
        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 0.7;
        self.visual.bloom_threshold = 0.3;
        self.visual.bloom_radius = 15.0;
        self.visual.vignette_strength = 0.4;
        self.visual.motion_blur = 0.1;
        self.visual.echo_enabled = false;

        // Audio settings
        self.audio.smoothing = 0.5;
        self.audio.beat_attack = 0.6;
        self.audio.beat_decay = 0.3;

        self.spectrum.enabled = false;
        self.waveform.enabled = true;
        self.waveform.style = WaveformStyle::Circle;
        self.waveform.amplitude = 50.0;
    }

    /// Hypnotic Vortex preset - intense, fast-spinning death spiral
    /// Maximum visual impact with chromatic aberration and radial blur
    pub fn preset_hypnotic_vortex(&mut self) {
        // Neon color scheme for maximum contrast
        self.color_scheme_index = 4; // Neon

        self.particles = ParticleConfig {
            enabled: true,
            count: 1000, // Reduced from 3000
            min_size: 2.5,
            max_size: 6.0,
            speed: 0.8,
            trail_length: 20,
            mode: ParticleMode::DeathSpiral,
            shape: ParticleShape::Glow,
            blend_mode: BlendMode::Add,
            spread: 360.0,
            gravity: 0.0,
            size_variation: 0.4,
            breathing_scale: 0.3,
            orbit_speed: 0.15,
            beat_size_pulse: 2.5,
            glow_intensity: 0.9,
            damping: 2.0,
            volumetric_rendering: true,
            volumetric_steps: 24,
            ..Default::default()
        };

        // More aggressive spiral settings
        self.death_spiral = DeathSpiralConfig {
            ring_count: 8,
            inner_radius: 50.0,
            ring_spacing: 1.3,
            base_rotation_speed: 1.2,
            alternate_direction: true,
            audio_speed_influence: 1.0,
            audio_radius_influence: 0.5,
            trail_intensity: 0.95,
            particles_per_ring: 80,
            wave_amplitude: 0.3,
            wave_frequency: 6.0,
            spiral_tightness: 0.15,
            beat_pulse_strength: 0.7,
            follow_strength: 0.5,
        };

        // Long, intense trails
        self.trails = TrailConfig {
            enabled: true,
            max_length: 5, // Reduced from 30
            fade_speed: 3.0,
            spawn_rate: 20.0,
            width_scale: 0.3,
            glow_enabled: true,
            opacity: 0.4, // Reduced from 0.85
            ..Default::default()
        };

        // Heavily reduced, cleaner connections
        self.connections = ConnectionConfig {
            enabled: false, // DISABLED to prevent white blotch
            max_distance: 30.0,
            max_connections: 2,
            opacity: 0.2,
            thickness: 1.0,
            audio_reactive: true,
            gradient_enabled: true,
            ..Default::default()
        };

        // Cinematic effects
        self.cinematic = CinematicConfig {
            chromatic_enabled: true,
            chromatic_intensity: 0.4,
            chromatic_radial: true,
            grain_enabled: true,
            grain_intensity: 0.05,
            grain_size: 1.5,
            vignette_enabled: true,
            vignette_intensity: 0.5,
            vignette_softness: 0.6,
            letterbox_enabled: false,
            letterbox_ratio: 2.35,
            color_temperature: 0.0,
        };

        // Heavy bloom and effects
        self.visual.bloom_enabled = true;
        self.visual.bloom_intensity = 1.2;
        self.visual.bloom_threshold = 0.25;
        self.visual.bloom_radius = 25.0;
        self.visual.chromatic_aberration = 0.3;
        self.visual.radial_blur_enabled = true;
        self.visual.radial_blur_amount = 0.1;
        self.visual.vignette_strength = 0.5;
        self.visual.echo_enabled = true;
        self.visual.echo_zoom = 1.008;
        self.visual.echo_rotation = 0.01;
        self.visual.echo_alpha = 0.3;

        // Responsive audio
        self.audio.smoothing = 0.3;
        self.audio.beat_attack = 0.8;
        self.audio.beat_decay = 0.4;
        self.audio.beat_explosion_strength = 5.0;

        self.spectrum.enabled = false;
        self.waveform.enabled = false;
    }
}
