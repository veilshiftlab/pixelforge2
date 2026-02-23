//! Unified ONNX Session Management
//!
//! Provides centralized model loading, caching, and execution provider configuration.
//! Backend priority (highest to lowest): CUDA → DirectML → CoreML → CPU
//! The best available backend is selected automatically at runtime.

use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

// ── Execution Backend ────────────────────────────────────────────────────────

/// Hardware execution backend for ONNX Runtime.
/// Ordered by general performance capability (best first).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionBackend {
    /// NVIDIA GPU via CUDA — best performance, requires CUDA + cuDNN
    Cuda,
    /// AMD/Intel GPU on Windows via DirectML — good, no extra deps on Windows 10+
    DirectML,
    /// Apple Neural Engine / GPU on macOS/iOS via CoreML
    CoreML,
    /// CPU fallback — always available
    Cpu,
}

impl ExecutionBackend {
    /// Detect and return all available backends on the current system,
    /// sorted from best to worst. CPU is always last.
    pub fn detect() -> Vec<Self> {
        let mut backends = Vec::new();

        // CUDA: best on any platform that has an NVIDIA GPU + toolkit
        if Self::cuda_available() {
            backends.push(Self::Cuda);
            log::info!("[Backend] CUDA available — NVIDIA GPU will be used");
        }

        // DirectML: Windows-only, covers AMD/Intel/NVIDIA dGPUs
        #[cfg(target_os = "windows")]
        if !backends.contains(&Self::Cuda) {
            // Only fall back to DirectML if CUDA isn't available;
            // CUDA is strictly better for NVIDIA hardware.
            backends.push(Self::DirectML);
            log::info!("[Backend] DirectML available — Windows GPU acceleration enabled");
        }

        // CoreML: macOS / iOS only
        #[cfg(target_os = "macos")]
        {
            backends.push(Self::CoreML);
            log::info!("[Backend] CoreML available — Apple Silicon / ANE will be used");
        }

        // CPU is always the final fallback
        backends.push(Self::Cpu);

        log::info!(
            "[Backend] Priority order: {:?}",
            backends
                .iter()
                .map(|b| format!("{b:?}"))
                .collect::<Vec<_>>()
                .join(" → ")
        );

        backends
    }

    /// Returns the single best backend available on this system.
    pub fn best() -> Self {
        Self::detect().into_iter().next().unwrap_or(Self::Cpu)
    }

    /// Probe CUDA + cuDNN availability using environment variables and known
    /// install paths. Also ensures cuDNN bin is on PATH so ORT can load it.
    fn cuda_available() -> bool {
        // ── 1. Find CUDA toolkit ──────────────────────────────────────────
        let cuda_root = std::env::var("CUDA_PATH").ok().or_else(|| {
            let fallbacks: &[&str] = &[
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3",
                "/usr/local/cuda",
                "/usr/cuda",
            ];
            fallbacks
                .iter()
                .find(|p| std::path::Path::new(p).exists())
                .map(|p| p.to_string())
        });

        let Some(cuda_root) = cuda_root else {
            log::debug!("[Backend] CUDA_PATH not set and no CUDA install found — skipping CUDA");
            return false;
        };
        log::debug!("[Backend] CUDA toolkit found at: {}", cuda_root);

        // ── 2. Find cuDNN bin directory ───────────────────────────────────
        // cuDNN may live separately (CUDNN_PATH) or bundled inside CUDA toolkit.
        let cudnn_bin: Option<std::path::PathBuf> = if cfg!(target_os = "windows") {
            // The CUDA-aware cuDNN installer nests DLLs under:
            //   CUDNN_PATH\bin\<cuda_ver>\x64\cudnn*.dll  (e.g. bin\12.9\x64)
            // Older / manual installs put them directly under:
            //   CUDNN_PATH\bin\cudnn*.dll
            // Also check inside the CUDA toolkit bin as a last resort.
            let search_roots: Vec<std::path::PathBuf> = {
                let mut v = Vec::new();
                if let Ok(p) = std::env::var("CUDNN_PATH") {
                    v.push(std::path::PathBuf::from(p));
                }
                v.push(std::path::PathBuf::from(&cuda_root));
                v
            };

            // Walk up to 3 levels deep under each root looking for any cudnn*.dll
            fn find_cudnn_dir(root: &std::path::Path) -> Option<std::path::PathBuf> {
                // BFS over subdirectory levels: root, root/*, root/*/*, root/*/*/*
                let mut queue: Vec<std::path::PathBuf> = vec![root.to_path_buf()];
                for _ in 0..3 {
                    let mut next = Vec::new();
                    for dir in &queue {
                        if let Ok(entries) = std::fs::read_dir(dir) {
                            let has_dll = entries
                                .filter_map(|e| e.ok())
                                .any(|e| {
                                    let name = e.file_name().to_string_lossy().to_lowercase();
                                    let is_dll = name.starts_with("cudnn") && name.ends_with(".dll");
                                    if is_dll {
                                        log::debug!("[Backend] Found cuDNN DLL: {}", e.path().display());
                                    }
                                    is_dll
                                });
                            if has_dll {
                                return Some(dir.clone());
                            }
                            // Queue subdirs for next level
                            if let Ok(subdirs) = std::fs::read_dir(dir) {
                                for sub in subdirs.filter_map(|e| e.ok()) {
                                    if sub.path().is_dir() {
                                        next.push(sub.path());
                                    }
                                }
                            }
                        }
                    }
                    queue = next;
                }
                None
            }

            search_roots.iter().find_map(|root| find_cudnn_dir(root))
        } else {
            let from_env = std::env::var("CUDNN_PATH")
                .ok()
                .map(|p| std::path::PathBuf::from(p).join("lib64"));
            let bundled = std::path::PathBuf::from(&cuda_root).join("lib64");
            [from_env, Some(bundled)].into_iter().flatten().find(|d| d.exists())
        };

        let Some(cudnn_bin) = cudnn_bin else {
            log::warn!(
                "[Backend] CUDA found at {} but cuDNN not located — \
                 set CUDNN_PATH or install cuDNN into the CUDA directory",
                cuda_root
            );
            return false;
        };
        log::info!("[Backend] cuDNN found at: {}", cudnn_bin.display());

        // ── 3. Ensure cuDNN bin is on PATH so ORT can dlopen it ───────────
        if cfg!(target_os = "windows") {
            let cudnn_str = cudnn_bin.to_string_lossy().to_string();
            let current = std::env::var("PATH").unwrap_or_default();
            if !current.to_lowercase().contains(&cudnn_str.to_lowercase()) {
                std::env::set_var("PATH", format!("{};{}", cudnn_str, current));
                log::info!("[Backend] Added cuDNN bin to PATH: {}", cudnn_str);
            }
        }

        true
    }

    /// Register this backend as an execution provider on the given session builder.
    fn apply(
        &self,
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder> {
        use ort::ep;

        let builder = match self {
            Self::Cuda => builder.with_execution_providers([
                ep::CUDA::default().build(),
            ])?,

            Self::DirectML => {
                // Device 0 is often the iGPU on hybrid systems.
                // Default to adapter 1 (discrete GPU) on hybrid systems.
                let device_id = best_directml_device_id();
                log::info!("[DirectML] Selected adapter index: {}", device_id);
                builder.with_execution_providers([
                    ep::DirectML::default().with_device_id(device_id).build(),
                ])?
            }

            Self::CoreML => builder.with_execution_providers([
                ep::CoreML::default().build(),
            ])?,

            Self::Cpu => builder, // ORT defaults to CPU; nothing to register
        };

        Ok(builder)
    }
}

/// Heuristic: pick the DirectML adapter that isn't the primary display adapter
/// when a discrete GPU is present. Falls back to 0 if none found.
fn best_directml_device_id() -> i32 {
    // On most hybrid systems: 0 = iGPU (display), 1 = dGPU (compute)
    // Without a full DXGI enumeration we use env var override or default to 1
    // so that the discrete GPU is preferred over the integrated one.
    if let Ok(val) = std::env::var("PIXELFORGE_DIRECTML_DEVICE") {
        if let Ok(id) = val.parse::<i32>() {
            return id;
        }
    }
    // Default to adapter 1 (discrete GPU) — safe because DirectML silently
    // falls back to adapter 0 if adapter 1 doesn't exist.
    1
}

// ── Model Types ──────────────────────────────────────────────────────────────

/// Types of ML models supported by PixelForge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    FaceDetection,
    DepthEstimation,
    Segmentation,
}

impl ModelType {
    /// Expected input shape in NCHW format.
    pub fn input_shape(self) -> [i64; 4] {
        match self {
            Self::FaceDetection => [1, 3, 160, 160],
            Self::DepthEstimation => [1, 3, 384, 384],
            Self::Segmentation => [1, 3, 512, 512],
        }
    }
}

// ── Model Session ─────────────────────────────────────────────────────────────

/// A loaded ONNX model with its metadata.
pub struct ModelSession {
    /// Inner ORT session. Mutex because `run()` requires `&mut self`.
    pub session: std::sync::Mutex<ort::session::Session>,
    /// Name of the primary input tensor.
    pub input_name: String,
    /// Name of the primary output tensor.
    pub output_name: String,
    /// Input shape in NCHW format.
    pub input_shape: [i64; 4],
    /// Which model this session represents.
    pub model_type: ModelType,
    /// Which backend is actually running inference.
    pub backend: ExecutionBackend,
}

// ── Session Manager ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct CacheKey {
    path: PathBuf,
    model_type: ModelType,
}

/// Centralized ONNX session manager with automatic backend selection and caching.
pub struct SessionManager {
    cache: RwLock<HashMap<CacheKey, Arc<ModelSession>>>,
    backend: ExecutionBackend,
    num_threads: usize,
    optimization_level: GraphOptimizationLevel,
}

impl SessionManager {
    /// Create a manager that auto-detects and uses the best available backend.
    pub fn new() -> Result<Self> {
        Self::with_backend(ExecutionBackend::best())
    }

    /// Create a manager pinned to a specific backend.
    pub fn with_backend(backend: ExecutionBackend) -> Result<Self> {
        log::info!("[SessionManager] Initialized with backend: {:?}", backend);
        Ok(Self {
            cache: RwLock::new(HashMap::new()),
            backend,
            num_threads: num_cpus(),
            optimization_level: GraphOptimizationLevel::Level3,
        })
    }

    /// Override the intra-op thread count (defaults to logical CPU count).
    pub fn with_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// The backend this manager is using.
    pub fn backend(&self) -> ExecutionBackend {
        self.backend
    }

    /// Retrieve a cached session or load the model from disk.
    pub fn get_or_load(
        &self,
        model_path: &PathBuf,
        model_type: ModelType,
    ) -> Result<Arc<ModelSession>> {
        let key = CacheKey {
            path: model_path.clone(),
            model_type,
        };

        // Fast path: read lock
        if let Some(session) = self.cache.read().get(&key) {
            log::debug!("[Cache] HIT {:?}", model_type);
            return Ok(Arc::clone(session));
        }

        // Slow path: load and insert
        log::info!("[Cache] MISS {:?} — loading from disk", model_type);
        let session = self.load_session(model_path, model_type)?;

        self.cache.write().insert(key, Arc::clone(&session));
        log::debug!("[Cache] Stored {:?}", model_type);

        Ok(session)
    }

    /// Load a new ONNX session, trying backends in priority order and falling
    /// back gracefully if the preferred one fails at runtime.
    fn load_session(
        &self,
        model_path: &PathBuf,
        model_type: ModelType,
    ) -> Result<Arc<ModelSession>> {
        log::info!(
            "[Load] {:?} from \"{}\"",
            model_type,
            model_path.display()
        );

        // Try preferred backend first, then fall back down the priority list
        let backends = {
            let mut list = ExecutionBackend::detect();
            // Ensure the manager's chosen backend is always tried first
            list.retain(|b| b != &self.backend);
            list.insert(0, self.backend);
            list
        };

        let mut last_err: Option<anyhow::Error> = None;

        for backend in &backends {
            log::info!("[Load] Trying backend: {:?}", backend);

            match self.try_load_with_backend(model_path, model_type, *backend) {
                Ok(session) => {
                    log::info!("[Load] ✓ {:?} loaded on {:?}", model_type, backend);
                    return Ok(session);
                }
                Err(e) => {
                    log::warn!("[Load] {:?} failed: {:#}", backend, e);
                    last_err = Some(e);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("No backends available")))
            .context(format!("Failed to load {:?} on any backend", model_type))
    }

    fn try_load_with_backend(
        &self,
        model_path: &PathBuf,
        model_type: ModelType,
        backend: ExecutionBackend,
    ) -> Result<Arc<ModelSession>> {
        let base_builder = ort::session::builder::SessionBuilder::new()?
            .with_optimization_level(self.optimization_level)?
            .with_intra_threads(self.num_threads)?;

        let builder = backend.apply(base_builder)?;
        let session = builder
            .commit_from_file(model_path)
            .with_context(|| format!("ORT commit_from_file failed for {:?}", backend))?;

        let input_name = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();
        let input_shape = model_type.input_shape();

        log::info!(
            "[Load] input={} output={} shape={:?} backend={:?}",
            input_name, output_name, input_shape, backend
        );

        Ok(Arc::new(ModelSession {
            session: std::sync::Mutex::new(session),
            input_name,
            output_name,
            input_shape,
            model_type,
            backend,
        }))
    }

    /// Remove all cached sessions.
    pub fn clear_cache(&self) {
        self.cache.write().clear();
        log::info!("[Cache] Cleared");
    }

    /// Remove one model from the cache.
    pub fn unload(&self, model_path: &PathBuf, model_type: ModelType) {
        let key = CacheKey {
            path: model_path.clone(),
            model_type,
        };
        self.cache.write().remove(&key);
        log::debug!("[Cache] Unloaded {:?}", model_type);
    }

    /// Returns `(cached_model_count, estimated_vram_mb)`.
    pub fn cache_stats(&self) -> (usize, usize) {
        let count = self.cache.read().len();
        (count, count * 50) // rough 50 MB/model estimate
    }

    /// Check whether a model is currently cached.
    pub fn is_loaded(&self, model_path: &PathBuf, model_type: ModelType) -> bool {
        let key = CacheKey {
            path: model_path.clone(),
            model_type,
        };
        self.cache.read().contains_key(&key)
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new().expect("Failed to create SessionManager")
    }
}

// ── Global Instance ───────────────────────────────────────────────────────────

static SESSION_MANAGER: std::sync::OnceLock<Arc<SessionManager>> = std::sync::OnceLock::new();

/// Returns the process-wide session manager, creating it on first call.
pub fn global_session_manager() -> Arc<SessionManager> {
    SESSION_MANAGER
        .get_or_init(|| {
            Arc::new(SessionManager::new().expect("Failed to create global SessionManager"))
        })
        .clone()
}

/// Initialize the global session manager with an explicit backend.
/// Must be called before any `global_session_manager()` call to take effect.
pub fn init_global_session_manager(backend: ExecutionBackend) -> Result<Arc<SessionManager>> {
    let manager = Arc::new(SessionManager::with_backend(backend)?);
    // OnceLock: only the first call wins; subsequent calls are no-ops.
    let _ = SESSION_MANAGER.set(manager.clone());
    Ok(manager)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Best-effort logical CPU count; defaults to 4 if unavailable.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}