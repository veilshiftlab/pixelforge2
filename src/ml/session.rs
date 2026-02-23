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
        let mut backends = Vec::new();
        
        log::info!("=== GPU Backend Detection (Platform: {:?}) ===", std::env::consts::OS);

        if cfg!(target_os = "windows") {
            log::info!("→ Windows detected: Using DirectML (native Windows GPU support)");
            log::info!("[DirectML] ✓ Available on Windows - No external dependencies needed");
            backends.push(ExecutionBackend::DirectML);
            
            log::info!("[CUDA] Also checking if CUDA is available (fallback)...");
            if ExecutionBackend::check_cuda_available() {
                log::info!("[CUDA] ✓ CUDA toolkit found - Available as fallback");
                backends.push(ExecutionBackend::Cuda);
            } else {
                log::info!("[CUDA] ✗ CUDA not found");
            }
        } else if cfg!(target_os = "linux") {
            log::info!("→ Linux detected: Prioritizing CUDA > CPU");
            log::info!("[CUDA] Checking if CUDA toolkit is installed...");
            if ExecutionBackend::check_cuda_available() {
                log::info!("[CUDA] ✓ CUDA toolkit found");
                backends.push(ExecutionBackend::Cuda);
            } else {
                log::warn!("[CUDA] ✗ CUDA toolkit not found");
            }
        } else if cfg!(target_os = "macos") {
            log::info!("→ macOS detected: Using CoreML");
            log::info!("[CoreML] ✓ Available on macOS");
            backends.push(ExecutionBackend::CoreML);
        }

        // CPU is always available as fallback
        log::info!("[CPU] Always available as fallback");
        backends.push(ExecutionBackend::Cpu);
        
        log::info!("=== Available backends (in priority order): {:?} ===\n", backends);
        backends
    }

    /// Check if CUDA is available and retrieve cuDNN path
    fn check_cuda_available() -> bool {
        log::info!("  → Checking for CUDA toolkit and cuDNN...");
        
        // First check CUDNN_PATH env var (if set, we already have cuDNN configured)
        if let Ok(cudnn_path) = std::env::var("CUDNN_PATH") {
            log::warn!("    ✓ CUDNN_PATH env var found: {}", cudnn_path);
            let cudnn_bin = std::path::PathBuf::from(&cudnn_path).join("bin");
            if cudnn_bin.exists() {
                log::warn!("    ✓ cuDNN bin directory found: {}", cudnn_bin.display());
                // Add cuDNN to PATH for ORT runtime
                if let Ok(current_path) = std::env::var("PATH") {
                    let new_path = format!("{};{}", cudnn_bin.display(), current_path);
                    std::env::set_var("PATH", new_path);
                    log::info!("    ✓ Added cuDNN bin to PATH for ORT runtime");
                }
                return true;
            }
        }
        
        // Try to find CUDA toolkit in common locations
        let cuda_paths = if cfg!(target_os = "windows") {
            vec![
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
                "C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA",
            ]
        } else if cfg!(target_os = "linux") {
            vec!["/usr/local/cuda", "/opt/cuda"]
        } else {
            vec![]
        };

        for path in cuda_paths {
            if std::path::Path::new(path).exists() {
                log::info!("    ✓ CUDA toolkit found at: {}", path);
                
                // Check for cuDNN (critical for GPU inference)
                // Try standard NVIDIA cuDNN install location
                let cudnn_paths = if cfg!(target_os = "windows") {
                    vec![
                        "C:\\Program Files\\NVIDIA\\CUDNN\\v13.1\\bin",
                        "C:\\Program Files\\NVIDIA\\CUDNN\\v13.0\\bin",
                        "C:\\Program Files\\NVIDIA\\CUDNN\\v12.9\\bin",
                        "C:\\Program Files\\NVIDIA\\CUDNN\\v9.19\\bin",
                        "C:\\Program Files\\cuDNN\\bin",
                    ]
                } else if cfg!(target_os = "linux") {
                    vec![
                        "/usr/local/cuda/lib64",
                        "/opt/cuda/lib64",
                    ]
                } else {
                    vec![]
                };
                
                let mut cudnn_found = false;
                for cudnn_path in &cudnn_paths {
                    let cudnn_dir = std::path::Path::new(cudnn_path);
                    if cudnn_dir.exists() {
                        log::warn!("    ✓ cuDNN found at: {}", cudnn_path);
                        // Add to PATH for ORT runtime
                        if let Ok(current_path) = std::env::var("PATH") {
                            let new_path = format!("{};{}", cudnn_path, current_path);
                            std::env::set_var("PATH", new_path);
                            log::info!("    ✓ Added cuDNN bin to PATH for ORT runtime");
                        }
                        cudnn_found = true;
                        break;
                    }
                }
                
                if !cudnn_found {
                    log::error!("    ⚠ ⚠ WARNING: cuDNN NOT FOUND ⚠ ⚠");
                    log::error!("      CUDA toolkit found but cuDNN is missing!");
                    log::error!("      GPU acceleration will NOT work without cuDNN");
                    log::error!("      Set CUDNN_PATH env var or install from: https://developer.nvidia.com/cudnn");
                }
                
                return true;
            }
        }

        // Also check via environment variable
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            log::info!("    ✓ CUDA_PATH env var found: {}", cuda_path);
            return true;
        }
        if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
            log::info!("    ✓ CUDA_HOME env var found: {}", cuda_home);
            return true;
        }

        log::warn!("  ✗ CUDA toolkit NOT found in common locations or env vars");
        false
    }

    /// Check if this backend is available
    pub fn is_available(&self) -> bool {
        let available = match self {
            ExecutionBackend::Cpu => {
                log::debug!("[CPU] Checking availability - always true");
                true
            }
            ExecutionBackend::Cuda => {
                let ok = Self::check_cuda_available();
                log::debug!("[CUDA] Checking availability - {}", if ok { "OK" } else { "FAIL" });
                ok
            }
            ExecutionBackend::DirectML => {
                let ok = cfg!(target_os = "windows");
                log::debug!("[DirectML] Checking availability on {} - {}", std::env::consts::OS, if ok { "OK" } else { "FAIL" });
                ok
            }
            ExecutionBackend::CoreML => {
                let ok = cfg!(target_os = "macos");
                log::debug!("[CoreML] Checking availability on {} - {}", std::env::consts::OS, if ok { "OK" } else { "FAIL" });
                ok
            }
        };
        available
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
    /// Create a new session manager, automatically selecting the best available backend
    pub fn new() -> Result<Self> {
        log::info!("\n┌────────────────────────────────────────┐");
        log::info!("│  SessionManager Initialization         │");
        log::info!("└────────────────────────────────────────┘");
        
        // Try to use GPU if available, otherwise fall back to CPU
        let backends = ExecutionBackend::available_backends();

        log::info!("\n>>> Attempting backends in priority order:");
        for (idx, backend) in backends.iter().enumerate() {
            log::info!("  [{}] {:?}", idx + 1, backend);
        }
        
        for backend in backends {
            if backend.is_available() && backend != ExecutionBackend::Cpu {
                log::info!("\n>>> Attempting to initialize: {:?}", backend);
                log::info!("    Status: Backend is available");
                match Self::with_backend(backend) {
                    Ok(manager) => {
                        log::warn!("\n✓✓✓ SUCCESS - Using {:?} backend ✓✓✓\n", backend);
                        return Ok(manager);
                    }
                    Err(e) => {
                        log::warn!("✗ Failed to initialize {:?}: {}", backend, e);
                        log::info!("    Trying next backend...");
                    }
                }
            } else {
                log::info!(">>> Skipping {:?} (backend not available or is CPU)", backend);
            }
        }

        // Fall back to CPU
        log::warn!("\n>>> All GPU backends unavailable or failed - Falling back to CPU\n");
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
                log::debug!("[CACHE HIT] Model already loaded: {:?}", model_type);
                return Ok(Arc::clone(session));
            }
        }

        log::info!("[LOADING] Model: {:?} from cache MISS", model_type);
        
        // Not in cache, create new session
        let session = self.create_session(model_path, model_type)?;

        // Store in cache (write lock)
        {
            let mut cache = self.cache.write();
            cache.insert(key, Arc::clone(&session));
            log::debug!("[CACHE] Stored {:?} in session cache", model_type);
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
            "\n╔═══════════════════════════════════════╗"
        );
        log::info!(
            "║ Creating ORT Session                  ║"
        );
        log::info!(
            "╠═══════════════════════════════════════╣"
        );
        log::info!(
            "║ Model Type:   {:?}",
            model_type
        );
        log::info!(
            "║ Path:         {:?}\n║",
            model_path
        );
        log::info!(
            "║ Backend:      {:?}",
            self.backend
        );
        log::info!(
            "║ Threads:      {}",
            self.num_threads
        );
        log::info!(
            "╚═══════════════════════════════════════╝\n"
        );

        let builder = ort::session::builder::SessionBuilder::new()?
            .with_optimization_level(self.optimization_level)?
            .with_intra_threads(self.num_threads)?;

        log::info!("→ Backend selected: {:?}", self.backend);
        
        // Build execution providers list based on selected backend
        let providers: Vec<ort::ep::ExecutionProviderDispatch> = match self.backend {
            ExecutionBackend::Cuda => {
                log::info!("  → Registering CUDA execution provider");
                log::info!("    Requirements: NVIDIA CUDA toolkit + cuDNN");
                vec![ort::ep::CUDA::default().build()]
            }
            ExecutionBackend::DirectML => {
                log::info!("  → Registering DirectML execution provider");
                log::info!("    Requirements: Windows 10/11 + GPU drivers");
                vec![ort::ep::DirectML::default().build()]
            }
            ExecutionBackend::CoreML => {
                log::info!("  → Registering CoreML execution provider");
                vec![ort::ep::CoreML::default().build()]
            }
            ExecutionBackend::Cpu => {
                log::warn!("  ⚠ CPU-only execution (no GPU provider)");
                vec![] // Empty = CPU only
            }
        };

        log::info!("\n→ Creating ONNX Runtime session from: {:?}", model_path);
        
        let builder = builder.with_execution_providers(providers)?;
        log::info!("✓ Execution providers registered successfully");
        
        let session = match builder.commit_from_file(model_path) {
            Ok(sess) => {
                log::warn!("✓ Session created successfully on registered provider");
                sess
            }
            Err(e) => {
                log::error!("✗ SESSION CREATION FAILED!");
                log::error!("  Error: {}", e);
                return Err(anyhow::anyhow!("Session creation error: {}", e));
            }
        };

        log::info!("\n→ Reading model metadata...");
        // Get input/output names
        let input_name = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();

        // Get input shape - use defaults based on model type
        let input_shape: Vec<i64> = match model_type {
            ModelType::FaceDetection => vec![1, 3, 160, 160],
            ModelType::DepthEstimation => vec![1, 3, 384, 384],
            ModelType::Segmentation => vec![1, 3, 512, 512],
        };

        log::warn!(
            "✓✓✓ Session Ready ✓✓✓\n  Backend:  {:?}\n  Input:    {} {:?}\n  Output:   {}\n",
            self.backend,
            input_name,
            input_shape,
            output_name
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
