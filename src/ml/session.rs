//! Unified ONNX Session Management
//!
//! Provides centralized model loading, caching, and GPU execution provider configuration.

use anyhow::Result;
use ort::session::builder::GraphOptimizationLevel;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Execution backend for ONNX Runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionBackend {
    /// CPU execution (default)
    #[default]
    Cpu,
    /// CUDA (NVIDIA GPU)
    Cuda,
    /// DirectML (Windows - AMD/Intel GPU)
    DirectML,
    /// CoreML (macOS)
    CoreML,
}

impl ExecutionBackend {
    /// Get available backends on this system
    pub fn available_backends() -> Vec<ExecutionBackend> {
        // Currently only CPU is always available
        // GPU backends require feature flags to be enabled
        vec![ExecutionBackend::Cpu]
    }

    /// Check if this backend is available
    pub fn is_available(&self) -> bool {
        matches!(self, ExecutionBackend::Cpu)
    }
}

/// Types of ML models supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    FaceDetection,
    DepthEstimation,
    Segmentation,
}

/// Cached ONNX session with metadata
pub struct ModelSession {
    /// The ONNX session (wrapped in Mutex since run() needs &mut self)
    pub session: std::sync::Mutex<ort::session::Session>,
    /// Input name for the model
    pub input_name: String,
    /// Output name for the model
    pub output_name: String,
    /// Input shape (NCHW format)
    pub input_shape: Vec<i64>,
    /// Model type identifier
    pub model_type: ModelType,
}

/// Model cache key
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct ModelKey {
    path: PathBuf,
    model_type: ModelType,
}

/// Centralized session manager for ONNX models
pub struct SessionManager {
    /// Cached sessions
    cache: RwLock<HashMap<ModelKey, Arc<ModelSession>>>,
    /// Execution backend
    backend: ExecutionBackend,
    /// Number of intra-op threads
    num_threads: usize,
    /// Graph optimization level
    optimization_level: GraphOptimizationLevel,
}

impl SessionManager {
    /// Create a new session manager with CPU backend
    pub fn new() -> Result<Self> {
        Self::with_backend(ExecutionBackend::Cpu)
    }

    /// Create a new session manager with specified backend
    pub fn with_backend(backend: ExecutionBackend) -> Result<Self> {
        Ok(Self {
            cache: RwLock::new(HashMap::new()),
            backend,
            num_threads: 4,
            optimization_level: GraphOptimizationLevel::Level3,
        })
    }

    /// Set number of threads for intra-op parallelism
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Get the current execution backend
    pub fn backend(&self) -> ExecutionBackend {
        self.backend
    }

    /// Get or create a session for the given model
    pub fn get_or_load(
        &self,
        model_path: &PathBuf,
        model_type: ModelType,
    ) -> Result<Arc<ModelSession>> {
        let key = ModelKey {
            path: model_path.clone(),
            model_type,
        };

        // Check cache first (read lock)
        {
            let cache = self.cache.read();
            if let Some(session) = cache.get(&key) {
                return Ok(Arc::clone(session));
            }
        }

        // Not in cache, create new session
        let session = self.create_session(model_path, model_type)?;

        // Store in cache (write lock)
        {
            let mut cache = self.cache.write();
            cache.insert(key, Arc::clone(&session));
        }

        Ok(session)
    }

    /// Create a new ONNX session
    fn create_session(
        &self,
        model_path: &PathBuf,
        model_type: ModelType,
    ) -> Result<Arc<ModelSession>> {
        log::info!(
            "Loading ONNX model: {:?} (type: {:?}, backend: {:?})",
            model_path,
            model_type,
            self.backend
        );

        let builder = ort::session::builder::SessionBuilder::new()?
            .with_optimization_level(self.optimization_level)?
            .with_intra_threads(self.num_threads)?;

        let session = builder.commit_from_file(model_path)?;

        // Get input/output names
        let input_name = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();

        // Get input shape - use defaults based on model type
        // Note: ort 2.0 doesn't expose shape info easily before inference
        // We use known shapes for our supported models
        let input_shape: Vec<i64> = match model_type {
            ModelType::FaceDetection => vec![1, 3, 160, 160],
            ModelType::DepthEstimation => vec![1, 3, 384, 384],
            ModelType::Segmentation => vec![1, 3, 512, 512],
        };

        log::info!(
            "Model loaded: input={}, output={}, shape={:?}",
            input_name,
            output_name,
            input_shape
        );

        Ok(Arc::new(ModelSession {
            session: std::sync::Mutex::new(session),
            input_name,
            output_name,
            input_shape,
            model_type,
        }))
    }

    /// Clear the model cache (unload all models)
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write();
        cache.clear();
        log::info!("Model cache cleared");
    }

    /// Remove a specific model from cache
    pub fn unload(&self, model_path: &PathBuf, model_type: ModelType) {
        let key = ModelKey {
            path: model_path.clone(),
            model_type,
        };
        let mut cache = self.cache.write();
        cache.remove(&key);
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read();
        let count = cache.len();
        // Estimate memory usage (rough approximation)
        let estimated_mb = count * 50; // Assume ~50MB per model
        (count, estimated_mb)
    }

    /// Check if a model is loaded in cache
    pub fn is_loaded(&self, model_path: &PathBuf, model_type: ModelType) -> bool {
        let key = ModelKey {
            path: model_path.clone(),
            model_type,
        };
        let cache = self.cache.read();
        cache.contains_key(&key)
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new().expect("Failed to create SessionManager")
    }
}

/// Global session manager instance (lazy initialized)
static SESSION_MANAGER: std::sync::OnceLock<Arc<SessionManager>> = std::sync::OnceLock::new();

/// Get or create the global session manager
pub fn global_session_manager() -> Arc<SessionManager> {
    SESSION_MANAGER
        .get_or_init(|| Arc::new(SessionManager::new().expect("Failed to create global SessionManager")))
        .clone()
}

/// Initialize the global session manager with a specific backend
pub fn init_global_session_manager(backend: ExecutionBackend) -> Result<Arc<SessionManager>> {
    let manager = Arc::new(SessionManager::with_backend(backend)?);
    // Note: OnceLock doesn't allow overwriting, so this only works on first call
    let _ = SESSION_MANAGER.set(manager.clone());
    Ok(manager)
}
