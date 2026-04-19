//! Runtime CPU feature detection.
//!
//! At present this is informational only — the stable `chunked` kernels
//! already compile down to AVX2 instructions when the Rust default
//! target (`x86-64-v3` is the usual Gentoo-ish choice) supports them,
//! and adding a hand-rolled `std::arch::x86_64::*` path on top has
//! shown no measurable gain in the microbenchmarks.
//!
//! The hooks are kept so that future kernels (for example an FFT-based
//! IMDCT that benefits from AVX-512) can query the runtime capabilities
//! and flip a function pointer without reshaping the dispatch API.

/// Description of the CPU we are running on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub fma: bool,
    pub neon: bool,
}

/// Detect features once at process start-up. Cheap to call repeatedly
/// — the result is cached after the first invocation.
pub fn detect() -> CpuFeatures {
    use std::sync::OnceLock;
    static CACHE: OnceLock<CpuFeatures> = OnceLock::new();
    *CACHE.get_or_init(detect_inner)
}

#[cfg(target_arch = "x86_64")]
fn detect_inner() -> CpuFeatures {
    CpuFeatures {
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        fma: std::arch::is_x86_feature_detected!("fma"),
        neon: false,
    }
}

#[cfg(target_arch = "aarch64")]
fn detect_inner() -> CpuFeatures {
    CpuFeatures {
        avx2: false,
        fma: false,
        neon: std::arch::is_aarch64_feature_detected!("neon"),
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_inner() -> CpuFeatures {
    CpuFeatures::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_runs() {
        // Smoke test: repeated calls return the same cached struct.
        let a = detect();
        let b = detect();
        assert_eq!(a, b);
    }
}
