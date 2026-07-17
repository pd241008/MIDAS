use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct MidasConfig {
    pub W: usize,
    pub D: usize,
    pub gamma: f32,
    pub lambda: f32,
    pub k: f32,
    pub tau: f32,
    pub delta_theta_max_deg: f32,
    pub sla_budget_ms: u64,
    pub channel_capacity: usize,
    pub model_path: String,
    pub manifold_path: String,
}

#[derive(Debug)]
pub enum ConfigError {
    OutOfRange(String),
    Io(String),
    Parse(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::OutOfRange(msg) => write!(f, "config out of range: {}", msg),
            ConfigError::Io(msg) => write!(f, "config I/O error: {}", msg),
            ConfigError::Parse(msg) => write!(f, "config parse error: {}", msg),
        }
    }
}

impl std::error::Error for ConfigError {}

impl MidasConfig {
    pub fn from_file(path: &str) -> Result<Self, ConfigError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| ConfigError::Io(e.to_string()))?;
        let config: MidasConfig =
            serde_json::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.W == 0 {
            return Err(ConfigError::OutOfRange(
                "W (window size) must be > 0".into(),
            ));
        }
        if self.D == 0 {
            return Err(ConfigError::OutOfRange(
                "D (feature dimension) must be > 0".into(),
            ));
        }
        if !(0.0..=2.0).contains(&self.gamma) {
            return Err(ConfigError::OutOfRange(
                "gamma must be in [0, 2]".into(),
            ));
        }
        if self.lambda <= 0.0 || self.lambda > 10.0 {
            return Err(ConfigError::OutOfRange(
                "lambda must be in (0, 10]".into(),
            ));
        }
        if self.k <= 0.0 || self.k > 10.0 {
            return Err(ConfigError::OutOfRange("k must be in (0, 10]".into()));
        }
        if !(0.0..=1.0).contains(&self.tau) {
            return Err(ConfigError::OutOfRange(
                "tau must be in [0, 1]".into(),
            ));
        }
        if self.delta_theta_max_deg <= 0.0 || self.delta_theta_max_deg > 180.0 {
            return Err(ConfigError::OutOfRange(
                "delta_theta_max_deg must be in (0, 180]".into(),
            ));
        }
        if self.sla_budget_ms == 0 || self.sla_budget_ms > 10000 {
            return Err(ConfigError::OutOfRange(
                "sla_budget_ms must be in (0, 10000]".into(),
            ));
        }
        if self.channel_capacity == 0 || self.channel_capacity > 256 {
            return Err(ConfigError::OutOfRange(
                "channel_capacity must be in (0, 256]".into(),
            ));
        }
        if self.model_path.is_empty() {
            return Err(ConfigError::OutOfRange("model_path must not be empty".into()));
        }
        if self.manifold_path.is_empty() {
            return Err(ConfigError::OutOfRange(
                "manifold_path must not be empty".into(),
            ));
        }
        Ok(())
    }

    pub fn delta_theta_max_rad(&self) -> f32 {
        self.delta_theta_max_deg * std::f32::consts::PI / 180.0
    }
}
