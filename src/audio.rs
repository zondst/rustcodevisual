//! Audio Analysis System for Particle Studio RS
//! Real FFT-based audio analysis with beat detection

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use rodio::{Decoder, OutputStream, Sink};
use rustfft::{num_complex::Complex, FftPlanner};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Audio frame data for visualization
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct AudioFrame {
    pub beat: f32,
    pub beat_intensity: f32,
    pub tempo: f32,
    pub bass: f32,
    pub mid: f32,
    pub high: f32,
    pub sub_bass: f32,
    pub amplitude: f32,
    pub spectral_flux: f32,
    pub spectrum: Vec<f32>,
    pub waveform: Vec<f32>,
    pub time: f32,
    pub frame_index: usize,
}

impl Default for AudioFrame {
    fn default() -> Self {
        Self {
            beat: 0.0,
            beat_intensity: 0.0,
            tempo: 120.0,
            bass: 0.0,
            mid: 0.0,
            high: 0.0,
            sub_bass: 0.0,
            amplitude: 0.0,
            spectral_flux: 0.0,
            spectrum: vec![0.0; 64],
            waveform: vec![0.0; 512],
            time: 0.0,
            frame_index: 0,
        }
    }
}

/// Pre-analyzed audio data for frame-accurate sync
#[derive(Clone)]
pub struct AudioAnalysis {
    pub frames: Vec<AudioFrame>,
    pub sample_rate: u32,
    pub duration: f32,
    pub total_frames: usize,
    pub fps: u32,
}

impl AudioAnalysis {
    pub fn get_frame(&self, index: usize) -> AudioFrame {
        self.frames.get(index).cloned().unwrap_or_default()
    }
}

/// Professional beat detector with decay/trigger/peak-valley detection
/// Ported from vis-core sample project
#[allow(dead_code)]
pub struct BeatDetector {
    decay: f32,   // How fast quiet beats can be detected (0.9995 = slow decay)
    trigger: f32, // Min peak-to-valley ratio to trigger beat (0.4 = 40%)
    last_volume: f32,
    last_delta: f32,
    last_beat_delta: f32,
    last_peak: f32,
    last_valley: f32,
}

impl BeatDetector {
    #[allow(dead_code)]
    pub fn new(decay: f32, trigger: f32) -> Self {
        Self {
            decay: 1.0 - 1.0 / decay, // Convert decay time to multiplier
            trigger,
            last_volume: 0.0,
            last_delta: 0.0,
            last_beat_delta: 0.0,
            last_peak: 0.0,
            last_valley: 0.0,
        }
    }

    pub fn default_audio() -> Self {
        Self::new(2000.0, 0.4) // Default: 2000 decay, 40% trigger
    }

    /// Detect if current volume represents a beat
    /// Returns (is_beat, beat_strength)
    pub fn detect(&mut self, volume: f32) -> (bool, f32) {
        // Decay beat_delta to allow quieter beats to be detected
        self.last_beat_delta *= self.decay;
        let delta = volume - self.last_volume;

        let (is_beat, strength) = if delta < 0.0 && self.last_delta > 0.0 {
            // Peak detected (volume started decreasing)
            self.last_peak = self.last_volume;
            let beat_delta = self.last_peak - self.last_valley;

            // Check if the peak is big enough relative to recent history
            if beat_delta > self.last_beat_delta * self.trigger {
                self.last_beat_delta = self.last_beat_delta.max(beat_delta);
                let strength = (beat_delta / self.last_beat_delta.max(0.001)).clamp(0.0, 1.0);
                (true, strength)
            } else {
                (false, 0.0)
            }
        } else if delta > 0.0 && self.last_delta < 0.0 {
            // Valley detected (volume started increasing)
            self.last_valley = self.last_volume;
            (false, 0.0)
        } else {
            (false, 0.0)
        };

        self.last_volume = volume;
        // Only write delta if the last two volumes weren't the same
        if delta.abs() > 0.001 {
            self.last_delta = delta;
        }

        (is_beat, strength)
    }

    /// Get the last measured volume
    pub fn last_volume(&self) -> f32 {
        self.last_volume
    }
}
/// Smooth exponential interpolation helper
fn interpolate_smooth(current: &mut f32, target: f32, rate: f32, dt: f32) {
    let smooth = 1.0 - (-rate * dt).exp();
    *current += smooth * (target - *current);
}

/// Normalized audio data after adaptive processing
#[derive(Clone, Debug)]
pub struct NormalizedAudio {
    pub bass: f32,                   // 0.0-1.0 normalized bass
    pub mid: f32,                    // 0.0-1.0 normalized mid
    pub high: f32,                   // 0.0-1.0 normalized high
    pub intensity: f32,              // Combined weighted intensity
    pub has_significant_audio: bool, // Above local average
    pub has_bass_hit: bool,          // Significant bass spike
    pub has_vocal: bool,             // Mid-range spike (vocals)
}

impl Default for NormalizedAudio {
    fn default() -> Self {
        Self {
            bass: 0.0,
            mid: 0.0,
            high: 0.0,
            intensity: 0.0,
            has_significant_audio: false,
            has_bass_hit: false,
            has_vocal: false,
        }
    }
}

/// Adaptive audio normalizer using sliding window averaging
/// Automatically adjusts to track dynamics - quiet tracks get boosted, loud tracks get tamed
pub struct AdaptiveAudioNormalizer {
    // History buffers for sliding window (about 3 seconds at 30fps = ~90 samples)
    bass_history: Vec<f32>,
    mid_history: Vec<f32>,
    high_history: Vec<f32>,

    // Current running statistics
    bass_avg: f32,
    mid_avg: f32,
    high_avg: f32,
    bass_peak: f32,
    mid_peak: f32,
    high_peak: f32,

    // Configuration
    window_size: usize,
    peak_decay: f32,
}

impl AdaptiveAudioNormalizer {
    pub fn new(window_seconds: f32, fps: u32) -> Self {
        let window_size = (window_seconds * fps as f32) as usize;
        Self {
            bass_history: Vec::with_capacity(window_size),
            mid_history: Vec::with_capacity(window_size),
            high_history: Vec::with_capacity(window_size),
            bass_avg: 0.0,
            mid_avg: 0.0,
            high_avg: 0.0,
            bass_peak: 0.1, // Start with small non-zero to avoid division issues
            mid_peak: 0.1,
            high_peak: 0.1,
            window_size: window_size.max(30), // At least 1 second
            peak_decay: 0.995,                // Slow peak decay
        }
    }

    /// Update with new audio values and get normalized output
    /// strength: 0.0-2.0, controls how much raw audio bleeds through vs normalized
    pub fn normalize(
        &mut self,
        bass: f32,
        mid: f32,
        high: f32,
        bass_sensitivity: f32,
        mid_sensitivity: f32,
        high_sensitivity: f32,
        strength: f32, // NEW: 0.0 = raw audio, 1.0 = fully adaptive, 2.0 = hyper-normalized
    ) -> NormalizedAudio {
        // Add to history
        self.bass_history.push(bass);
        self.mid_history.push(mid);
        self.high_history.push(high);

        // Trim to window size
        if self.bass_history.len() > self.window_size {
            self.bass_history.remove(0);
            self.mid_history.remove(0);
            self.high_history.remove(0);
        }

        // Calculate running averages
        let bass_sum: f32 = self.bass_history.iter().sum();
        let mid_sum: f32 = self.mid_history.iter().sum();
        let high_sum: f32 = self.high_history.iter().sum();
        let count = self.bass_history.len() as f32;

        self.bass_avg = bass_sum / count;
        self.mid_avg = mid_sum / count;
        self.high_avg = high_sum / count;

        // Update peaks with decay
        self.bass_peak = (self.bass_peak * self.peak_decay)
            .max(bass)
            .max(self.bass_avg * 1.5);
        self.mid_peak = (self.mid_peak * self.peak_decay)
            .max(mid)
            .max(self.mid_avg * 1.5);
        self.high_peak = (self.high_peak * self.peak_decay)
            .max(high)
            .max(self.high_avg * 1.5);

        // Normalize: (value - avg) / (peak - avg)
        let bass_range = (self.bass_peak - self.bass_avg).max(0.01);
        let mid_range = (self.mid_peak - self.mid_avg).max(0.01);
        let high_range = (self.high_peak - self.high_avg).max(0.01);

        // Normalized values (relative to local context)
        let bass_normalized = ((bass - self.bass_avg) / bass_range).clamp(0.0, 1.5);
        let mid_normalized = ((mid - self.mid_avg) / mid_range).clamp(0.0, 1.5);
        let high_normalized = ((high - self.high_avg) / high_range).clamp(0.0, 1.5);

        // Raw values scaled up for visibility
        let bass_raw = (bass * 5.0).clamp(0.0, 1.0);
        let mid_raw = (mid * 5.0).clamp(0.0, 1.0);
        let high_raw = (high * 5.0).clamp(0.0, 1.0);

        // BLEND: strength controls mix between raw and normalized
        // strength=0 -> 100% raw, strength=1 -> 50/50, strength=2 -> 100% normalized
        let blend = (strength * 0.5).clamp(0.0, 1.0);
        let bass_blended = bass_raw * (1.0 - blend) + bass_normalized * blend;
        let mid_blended = mid_raw * (1.0 - blend) + mid_normalized * blend;
        let high_blended = high_raw * (1.0 - blend) + high_normalized * blend;

        // Apply sensitivity
        let bass_norm = bass_blended * bass_sensitivity;
        let mid_norm = mid_blended * mid_sensitivity;
        let high_norm = high_blended * high_sensitivity;

        // Combined intensity with frequency weighting
        let intensity = bass_norm * 0.5 + mid_norm * 0.35 + high_norm * 0.15;

        // Detection thresholds - LESS STRICT for better reactivity
        // Use raw audio for detection, not normalized (so silence is truly silent)
        let raw_total = bass + mid + high;
        let has_bass_hit = bass > self.bass_avg * 1.2 && bass > 0.02;
        let has_vocal = mid > self.mid_avg * 1.2 && mid > 0.02;
        // has_significant_audio: true if there's any audible sound
        let has_significant_audio = raw_total > 0.01 || intensity > 0.1;

        NormalizedAudio {
            bass: bass_norm.clamp(0.0, 1.0),
            mid: mid_norm.clamp(0.0, 1.0),
            high: high_norm.clamp(0.0, 1.0),
            intensity: intensity.clamp(0.0, 1.0),
            has_significant_audio,
            has_bass_hit,
            has_vocal,
        }
    }

    /// Reset the normalizer (e.g., when loading new track)
    pub fn reset(&mut self) {
        self.bass_history.clear();
        self.mid_history.clear();
        self.high_history.clear();
        self.bass_avg = 0.0;
        self.mid_avg = 0.0;
        self.high_avg = 0.0;
        self.bass_peak = 0.1;
        self.mid_peak = 0.1;
        self.high_peak = 0.1;
    }
}

/// Real-time audio state for visualization
pub struct AudioState {
    // Raw values (instant)
    pub bass: f32,
    pub mid: f32,
    pub high: f32,
    pub amplitude: f32,
    pub beat: f32,
    pub spectrum: Vec<f32>,
    pub waveform: Vec<f32>,

    // Smoothed values for visualization (gradual)
    pub smooth_bass: f32,
    pub smooth_mid: f32,
    pub smooth_high: f32,
    pub smooth_amplitude: f32,
    pub smooth_beat: f32,

    // Smoothing
    prev_bass: f32,
    prev_mid: f32,
    prev_high: f32,
    prev_amplitude: f32,

    // Beat detection
    #[allow(dead_code)]
    beat_energy_history: Vec<f32>,
    #[allow(dead_code)]
    beat_threshold: f32,
}

impl AudioState {
    pub fn new() -> Self {
        Self {
            bass: 0.0,
            mid: 0.0,
            high: 0.0,
            amplitude: 0.0,
            beat: 0.0,
            spectrum: vec![0.0; 64],
            waveform: vec![0.0; 512],
            smooth_bass: 0.0,
            smooth_mid: 0.0,
            smooth_high: 0.0,
            smooth_amplitude: 0.0,
            smooth_beat: 0.0,
            prev_bass: 0.0,
            prev_mid: 0.0,
            prev_high: 0.0,
            prev_amplitude: 0.0,
            beat_energy_history: Vec::with_capacity(43),
            beat_threshold: 1.3,
        }
    }

    /// Update from pre-analyzed frame
    pub fn update_from_frame(&mut self, frame: &AudioFrame, smoothing: f32) {
        let s = smoothing.clamp(0.0, 0.99);

        self.bass = self.prev_bass * s + frame.bass * (1.0 - s);
        self.mid = self.prev_mid * s + frame.mid * (1.0 - s);
        self.high = self.prev_high * s + frame.high * (1.0 - s);
        self.amplitude = self.prev_amplitude * s + frame.amplitude * (1.0 - s);
        self.beat = frame.beat;

        self.spectrum = frame.spectrum.clone();
        self.waveform = frame.waveform.clone();

        self.prev_bass = self.bass;
        self.prev_mid = self.mid;
        self.prev_high = self.high;
        self.prev_amplitude = self.amplitude;
    }

    /// Update smoothed values with exponential interpolation
    /// Call this every frame for gradual transitions
    pub fn update_smoothing(
        &mut self,
        dt: f32,
        smoothing_rate: f32,
        beat_attack: f32,
        beat_decay: f32,
    ) {
        let rate = smoothing_rate * 8.0; // Scale for usable range

        // Smooth bass/mid/high/amplitude
        interpolate_smooth(&mut self.smooth_bass, self.bass, rate, dt);
        interpolate_smooth(&mut self.smooth_mid, self.mid, rate, dt);
        interpolate_smooth(&mut self.smooth_high, self.high, rate, dt);
        interpolate_smooth(&mut self.smooth_amplitude, self.amplitude, rate, dt);

        // Beat has asymmetric attack/decay
        if self.beat > self.smooth_beat {
            // Fast attack - quickly respond to beats
            let attack_blend = beat_attack.clamp(0.3, 1.0);
            self.smooth_beat = self.smooth_beat * (1.0 - attack_blend) + self.beat * attack_blend;
        } else {
            // Slow decay - gradually fade out
            let decay_rate = beat_decay * 3.0;
            interpolate_smooth(&mut self.smooth_beat, self.beat, decay_rate, dt);
        }
    }

    /// Update for demo mode (when no audio loaded) - minimal animation
    pub fn update_fake(&mut self, _dt: f32) {
        // When no audio is loaded, keep everything at low/zero values
        // This prevents particles from blinking/animating without music

        // Decay existing values slowly
        self.beat *= 0.95;
        self.bass *= 0.98;
        self.mid *= 0.98;
        self.high *= 0.98;
        self.amplitude *= 0.98;

        // Set to near-zero when below threshold
        if self.beat < 0.01 {
            self.beat = 0.0;
        }
        if self.bass < 0.01 {
            self.bass = 0.0;
        }
        if self.mid < 0.01 {
            self.mid = 0.0;
        }
        if self.high < 0.01 {
            self.high = 0.0;
        }
        if self.amplitude < 0.01 {
            self.amplitude = 0.0;
        }

        // Zero spectrum and flat waveform (no animation)
        for v in self.spectrum.iter_mut() {
            *v *= 0.95;
            if *v < 0.01 {
                *v = 0.0;
            }
        }

        for v in self.waveform.iter_mut() {
            *v *= 0.95;
        }
    }
}

/// Audio System - handles playback and analysis
pub struct AudioSystem {
    _stream: OutputStream,
    _stream_handle: rodio::OutputStreamHandle,
    sink: Arc<Sink>,
    pub analysis: Option<Arc<AudioAnalysis>>,
    pub audio_path: Option<String>,
    is_stopped: bool, // Track if stop was called (sink is empty)
}

impl AudioSystem {
    pub fn new() -> Self {
        let (_stream, stream_handle) =
            OutputStream::try_default().expect("Failed to create audio output stream");
        let sink = Arc::new(Sink::try_new(&stream_handle).expect("Failed to create sink"));

        Self {
            _stream,
            _stream_handle: stream_handle,
            sink,
            analysis: None,
            audio_path: None,
            is_stopped: false,
        }
    }

    /// Load and analyze audio file
    pub fn load_file(&mut self, path: String, fps: u32) -> anyhow::Result<()> {
        // Analyze audio first
        let analysis = analyze_audio(&path, fps)?;
        self.analysis = Some(Arc::new(analysis));
        self.audio_path = Some(path.clone());

        // Load for playback
        self.reload_audio_source()?;
        self.is_stopped = false;

        Ok(())
    }

    /// Reload audio source from stored path (internal use)
    fn reload_audio_source(&self) -> anyhow::Result<()> {
        if let Some(ref path) = self.audio_path {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let source = Decoder::new(reader)?;

            self.sink.stop();
            self.sink.append(source);
            self.sink.pause(); // Start paused, will play when play() is called
        }
        Ok(())
    }

    pub fn play(&mut self) {
        // If stopped, reload the audio source first
        if self.is_stopped || self.sink.empty() {
            if let Err(e) = self.reload_audio_source() {
                eprintln!("Failed to reload audio: {}", e);
                return;
            }
            self.is_stopped = false;
        }
        self.sink.play();
    }

    pub fn pause(&self) {
        self.sink.pause();
    }

    pub fn stop(&mut self) {
        self.sink.stop();
        self.is_stopped = true;
    }

    #[allow(dead_code)]
    pub fn is_playing(&self) -> bool {
        !self.sink.is_paused() && !self.sink.empty()
    }

    #[allow(dead_code)]
    pub fn set_volume(&self, volume: f32) {
        self.sink.set_volume(volume.clamp(0.0, 1.0));
    }

    /// Check if audio file is loaded
    pub fn is_loaded(&self) -> bool {
        self.analysis.is_some()
    }

    /// Get audio duration in seconds
    pub fn get_duration(&self) -> Option<f32> {
        self.analysis.as_ref().map(|a| a.duration)
    }
}

/// Analyze entire audio file and extract per-frame data
pub fn analyze_audio(path: &str, fps: u32) -> anyhow::Result<AudioAnalysis> {
    let path = Path::new(path);
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension() {
        hint.with_extension(&ext.to_string_lossy());
    }

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed =
        symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;

    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or_else(|| anyhow::anyhow!("No audio track found"))?;

    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(2);
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;

    // Collect all samples
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        match format.next_packet() {
            Ok(packet) => {
                if packet.track_id() != track_id {
                    continue;
                }

                match decoder.decode(&packet) {
                    Ok(decoded) => {
                        let spec = decoded.spec();
                        let frames = decoded.frames();

                        let mut sample_buf = SampleBuffer::<f32>::new(frames as u64, *spec);
                        sample_buf.copy_interleaved_ref(decoded);

                        // Mix to mono
                        let samples = sample_buf.samples();
                        for chunk in samples.chunks(channels) {
                            let mono: f32 = chunk.iter().sum::<f32>() / channels as f32;
                            all_samples.push(mono);
                        }
                    }
                    Err(_) => continue,
                }
            }
            Err(_) => break,
        }
    }

    let duration = all_samples.len() as f32 / sample_rate as f32;
    let samples_per_frame = sample_rate as usize / fps as usize;
    let total_frames = all_samples.len() / samples_per_frame;

    // FFT setup
    let fft_size = 2048;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Frequency band boundaries (Hz)
    let sub_bass_max = 60.0;
    let bass_max = 250.0;
    let mid_max = 2000.0;
    let high_max = 20000.0;

    let freq_resolution = sample_rate as f32 / fft_size as f32;

    let sub_bass_bin = (sub_bass_max / freq_resolution) as usize;
    let bass_bin = (bass_max / freq_resolution) as usize;
    let mid_bin = (mid_max / freq_resolution) as usize;
    let high_bin = ((high_max / freq_resolution) as usize).min(fft_size / 2);

    // Analyze each frame
    let mut frames = Vec::with_capacity(total_frames);
    let mut prev_spectrum = vec![0.0f32; fft_size / 2];
    let mut energy_history: Vec<f32> = Vec::new();

    for frame_idx in 0..total_frames {
        let start_sample = frame_idx * samples_per_frame;
        let end_sample = (start_sample + fft_size).min(all_samples.len());

        // Get samples for this frame
        let mut fft_buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); fft_size];

        for (i, &sample) in all_samples[start_sample..end_sample].iter().enumerate() {
            // Apply Hann window
            let window =
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos());
            fft_buffer[i] = Complex::new(sample * window, 0.0);
        }

        // Perform FFT
        fft.process(&mut fft_buffer);

        // Extract magnitude spectrum
        let mut spectrum: Vec<f32> = fft_buffer[..fft_size / 2]
            .iter()
            .map(|c| (c.norm() / fft_size as f32).sqrt())
            .collect();

        // Normalize
        let max_mag = spectrum.iter().cloned().fold(0.0f32, f32::max);
        if max_mag > 0.0 {
            for s in &mut spectrum {
                *s /= max_mag;
            }
        }

        // Extract frequency bands
        let sub_bass =
            spectrum[1..sub_bass_bin.max(2)].iter().sum::<f32>() / (sub_bass_bin - 1).max(1) as f32;
        let bass = spectrum[sub_bass_bin..bass_bin.max(sub_bass_bin + 1)]
            .iter()
            .sum::<f32>()
            / (bass_bin - sub_bass_bin).max(1) as f32;
        let mid = spectrum[bass_bin..mid_bin.max(bass_bin + 1)]
            .iter()
            .sum::<f32>()
            / (mid_bin - bass_bin).max(1) as f32;
        let high = spectrum[mid_bin..high_bin.max(mid_bin + 1)]
            .iter()
            .sum::<f32>()
            / (high_bin - mid_bin).max(1) as f32;

        // Spectral flux for beat detection
        let spectral_flux: f32 = spectrum
            .iter()
            .zip(prev_spectrum.iter())
            .map(|(c, p)| (c - p).max(0.0))
            .sum();

        // Beat detection
        energy_history.push(spectral_flux);
        if energy_history.len() > 43 {
            energy_history.remove(0);
        }

        let avg_energy = energy_history.iter().sum::<f32>() / energy_history.len() as f32;
        let beat = if spectral_flux > avg_energy * 1.3 {
            1.0
        } else {
            0.0
        };
        let beat_intensity = (spectral_flux / (avg_energy + 0.001)).min(2.0) - 1.0;

        // Amplitude
        let amplitude = (bass * 0.4 + mid * 0.35 + high * 0.25).min(1.0);

        // Waveform for this frame
        let waveform_samples = 512;
        let waveform: Vec<f32> = (0..waveform_samples)
            .map(|i| {
                let idx = start_sample + (i * samples_per_frame / waveform_samples);
                all_samples.get(idx).copied().unwrap_or(0.0)
            })
            .collect();

        // Resample spectrum to 64 bands
        let spectrum_bands = 64;
        let resampled_spectrum: Vec<f32> = (0..spectrum_bands)
            .map(|i| {
                let start = i * spectrum.len() / spectrum_bands;
                let end = ((i + 1) * spectrum.len() / spectrum_bands).max(start + 1);
                spectrum[start..end].iter().sum::<f32>() / (end - start) as f32
            })
            .collect();

        frames.push(AudioFrame {
            beat,
            beat_intensity: beat_intensity.max(0.0),
            tempo: 120.0, // TODO: tempo detection
            bass,
            mid,
            high,
            sub_bass,
            amplitude,
            spectral_flux,
            spectrum: resampled_spectrum,
            waveform,
            time: frame_idx as f32 / fps as f32,
            frame_index: frame_idx,
        });

        prev_spectrum = spectrum;
    }

    Ok(AudioAnalysis {
        frames,
        sample_rate,
        duration,
        total_frames,
        fps,
    })
}
