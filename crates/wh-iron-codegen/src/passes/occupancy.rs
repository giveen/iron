//! Copyright 2026 Eric Kryski (@ekryski), Tom Turney (@TheTom) and 0xClandestine (@0xClandestine)
//! SPDX-License-Identifier: Apache-2.0
//! Occupancy Estimation — predict GPU thread occupancy from register pressure.
//!
//! Computes an estimated occupancy percentage based on register pressure,
//! threadgroup memory usage, and threadgroup size.  This analysis feeds the
//! autotuner, not the compilation pipeline directly.
//!
//! ## Apple GPU Context
//!
//! | Family | Max Threads/TG | TG Memory | Reg Guide | Notes |
//! |---|---|---|---|---|
//! | Apple7 (M1) | 1024 | ~32KB | 128 | Fixed allocation |
//! | Apple8 (M2) | 1024 | ~32KB | 128 | Similar to M1 |
//! | Apple9 (M3) | 1024 | ~32KB | Dynamic | OMU-managed |
//! | Apple10 (M4) | 1024 | ~32KB | Dynamic | Improved OMU |
//! | Apple11 (M5) | 1024 | ~32KB | Dynamic | Smarter OMU |
//!
//! For M3+, register allocation is dynamically managed by the Occupancy
//! Management Unit (OMU).  Our 128-register guide is a soft heuristic —
//! the OMU may run shaders above or below this threshold depending on
//! cache pressure and available L1.  We model register pressure as a
//! gradual degradation, not a hard ceiling.
//!
//! ## Usage
//!
//! This module is not a [`Pass`] — it runs as post-pipeline analysis that
//! feeds into the autotuner.
//!
//! ## References
//! - Apple (2023), "Explore GPU advancements in M3 and A17 Pro",
//!   WWDC Tech Talks.  Introduced the OMU and dynamic register allocation.
//!   https://developer.apple.com/videos/play/tech-talks/111375
//! - Apple (2021), "Create image processing apps powered by Apple silicon",
//!   WWDC21.  Register pressure and occupancy trade-offs on Apple GPUs.
//!   https://developer.apple.com/videos/play/wwdc2021/10153/
//! - Apple, "Finding your Metal app's GPU occupancy", Xcode documentation.
//!   https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy
//! - Rosenzweig (2021), "Dissecting the Apple M1 GPU, part III",
//!   Asahi Linux blog.  Reverse-engineered register file and occupancy details.
//!   https://alyssarosenzweig.ca/blog/asahi-gpu-part-3/
//! - Poletto & Sarkar (1999), "Linear scan register allocation",
//!   ACM TOPLAS 21(5):895–913.  Foundation for the linear-scan liveness
//!   model used in [`register_estimate`].

use std::fmt;

use wh_iron_core::ir::Kernel;

use super::register_estimate;

/// Per-GPU-family resource limits.
#[derive(Debug, Clone, Copy)]
pub struct GpuLimits {
    /// Maximum threads per threadgroup.
    pub max_threads_per_tg: u32,
    /// Threadgroup memory in bytes.
    pub tg_memory_bytes: u32,
    /// Soft register guide (not a hard ceiling on M3+ where the OMU
    /// dynamically allocates registers).
    pub regs_per_thread_guide: u32,
}

impl Default for GpuLimits {
    fn default() -> Self {
        GpuLimits {
            max_threads_per_tg: 1024,
            tg_memory_bytes: 32 * 1024,
            regs_per_thread_guide: 128,
        }
    }
}

/// Bottleneck preventing higher occupancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    /// Register pressure is degrading occupancy.
    RegisterLimited,
    /// Threadgroup memory size is the limiting factor.
    MemoryLimited,
    /// Thread count is the limiting factor.
    ThreadLimited,
    /// Tile working set exceeds likely on-chip cache.
    /// The OMU may throttle occupancy to prevent L1 thrashing.
    /// (Set by the autotuner when tile dims are known; not computed in estimate_occupancy.)
    CachePressure,
}

impl fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Bottleneck::RegisterLimited => "register-limited",
            Bottleneck::MemoryLimited => "memory-limited",
            Bottleneck::ThreadLimited => "thread-limited",
            Bottleneck::CachePressure => "cache-pressure",
        })
    }
}

/// Occupancy estimate for a kernel with a given threadgroup size.
#[derive(Debug, Clone)]
pub struct OccupancyEstimate {
    /// Estimated occupancy as a percentage (0.0–100.0).
    pub occupancy_pct: f64,
    /// The primary bottleneck.
    pub bottleneck: Bottleneck,
    /// Upper bound on simultaneous threadgroups per shader core.
    ///
    /// Computed from the simple resource model. The actual count is decided
    /// by the OMU at runtime and may be lower due to cache pressure.
    pub max_tgs_per_cu: Option<u32>,
}

/// Compute an occupancy estimate for `kernel` with the given `threadgroup_size`.
///
/// `tg_mem_usage_bytes` is an optional estimate of threadgroup memory usage.
/// If None, memory is assumed not to be the bottleneck.
pub fn estimate_occupancy(
    kernel: &Kernel,
    threadgroup_size: u32,
    tg_mem_usage_bytes: Option<u32>,
) -> OccupancyEstimate {
    let limits = GpuLimits::default();
    let reg_est = register_estimate::estimate_registers(kernel);

    // --- Register pressure (soft degradation, not a hard ceiling) ---
    //
    // On M3+, the OMU dynamically allocates registers. High register usage
    // degrades occupancy gradually rather than hitting a hard cliff at 128.
    // At ≤ the guide value, no penalty. Beyond that, linear degradation
    // to 10% at double the guide (i.e., at 256 regs/thr).
    let reg_occ = if reg_est.regs_per_thread <= limits.regs_per_thread_guide as usize {
        1.0
    } else {
        let excess = reg_est.regs_per_thread as f64 - limits.regs_per_thread_guide as f64;
        (1.0 - excess / limits.regs_per_thread_guide as f64).max(0.1)
    };

    // --- Thread-limited occupancy ---
    //
    // Hard ceiling: max 1024 threads per threadgroup on Apple GPUs.
    let thr_occ = limits.max_threads_per_tg as f64 / threadgroup_size as f64;
    let thr_occ = thr_occ.min(1.0);

    // --- Threadgroup memory: multi-TG residency, not a per-TG ratio ---
    //
    // Threadgroup memory on Apple GPUs is a single ~32KB SRAM pool SHARED by
    // all threadgroups co-resident on a shader core — it is not a private
    // budget with headroom above one TG's own usage. A single TG's static
    // allocation can never legally exceed the pool (it wouldn't compile), so
    // a plain `pool / mem_used` ratio is always >= 1.0 and can never signal
    // a memory-bound kernel. Instead we compute how many TGs of this
    // footprint can be co-resident in the pool at once — a residency count,
    // exactly like the thread-count dimension below — and only THEN turn
    // that into an occupancy fraction relative to what the thread dimension
    // alone would allow.
    //
    // `None` (no TG-memory estimate supplied) or `Some(0)` (kernel uses no
    // threadgroup memory) both mean "memory does not constrain residency".
    let mem_limited_tgs: Option<u32> = match tg_mem_usage_bytes {
        None | Some(0) => None,
        Some(mem_used) => Some(limits.tg_memory_bytes / mem_used),
    };

    // --- Thread-count residency, as a TG count (not just a ratio) ---
    //
    // Reuses the same max-threads-per-TG hard limit as `thr_occ` above to
    // express "how many TGs of this size could be co-resident" purely from
    // a thread-count standpoint, so it's directly comparable to
    // `mem_limited_tgs`. For any legal threadgroup size (<= the hard per-TG
    // limit) this divides out to >= 1 automatically; a threadgroup size
    // that already exceeds the hard limit (invalid on its own terms) floors
    // to 0, correctly signaling "can't even fit one".
    let thread_limited_tgs = limits.max_threads_per_tg / threadgroup_size.max(1);

    // Fold memory residency back into an occupancy fraction: relative to the
    // number of TGs the thread dimension alone would allow, how many can
    // actually be resident once threadgroup memory is accounted for.
    let mem_occ = match mem_limited_tgs {
        None => 1.0,
        Some(m) => (m as f64 / thread_limited_tgs.max(1) as f64).min(1.0),
    };

    // Occupancy is the minimum across all dimensions.
    let mut occ = reg_occ.min(thr_occ).min(mem_occ);
    occ = (occ * 1000.0).round() / 1000.0;

    // --- Bottleneck identification ---
    //
    // Pick the strictest limiter. When multiple limiters are within rounding
    // tolerance of each other, we report the most actionable one (register
    // pressure > memory > thread count).
    let bottleneck = if occ >= 0.999 {
        Bottleneck::ThreadLimited
    } else if reg_occ <= mem_occ && reg_occ <= thr_occ || (reg_occ - occ).abs() < 0.002 {
        Bottleneck::RegisterLimited
    } else if mem_occ <= thr_occ || (mem_occ - occ).abs() < 0.002 {
        Bottleneck::MemoryLimited
    } else {
        Bottleneck::ThreadLimited
    };

    // Residency count: the real number of TGs that can be co-resident on a
    // core, taking BOTH dimensions into account — not derived from the
    // blended fraction above (which mixes in the soft register-pressure
    // signal and would give a misleadingly high count for memory-bound
    // kernels; e.g. a 20KiB/TG kernel at TG=256 has thr_occ==1.0 and
    // reg_occ==1.0, so 1/occ would report 4 residency slots when the true
    // memory-bound answer is 1).
    let max_tgs_per_cu = Some(match mem_limited_tgs {
        None => thread_limited_tgs,
        Some(m) => thread_limited_tgs.min(m),
    });

    OccupancyEstimate { occupancy_pct: occ * 100.0, bottleneck, max_tgs_per_cu }
}

/// Convenience: estimate occupancy for common threadgroup sizes and return the best.
///
/// `candidates` is a list of (threadgroup_size, tg_mem_bytes) to evaluate.
/// Returns the candidate with the highest estimated occupancy.
pub fn best_threadgroup_size(
    kernel: &Kernel,
    candidates: &[(u32, Option<u32>)],
) -> Option<(u32, OccupancyEstimate)> {
    let mut best: Option<(u32, OccupancyEstimate)> = None;

    for &(tg_size, mem) in candidates {
        let est = estimate_occupancy(kernel, tg_size, mem);
        match &best {
            None => best = Some((tg_size, est)),
            Some((_, prev)) if est.occupancy_pct > prev.occupancy_pct => {
                best = Some((tg_size, est));
            },
            _ => {},
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use wh_iron_core::ir::{Op, ValueId};

    use super::*;

    #[test]
    fn empty_kernel_full_occupancy() {
        let k = Kernel::new("empty");
        let est = estimate_occupancy(&k, 256, None);
        assert!((est.occupancy_pct - 100.0).abs() < 0.1);
        assert_eq!(est.bottleneck, Bottleneck::ThreadLimited);
    }

    #[test]
    fn no_tg_memory_is_not_memory_limited() {
        // A kernel that reports zero threadgroup-memory usage must not be
        // treated as memory-constrained (Some(0) behaves like None).
        let k = Kernel::new("no_shmem");
        let est = estimate_occupancy(&k, 256, Some(0));
        assert!((est.occupancy_pct - 100.0).abs() < 0.1);
        assert_eq!(est.bottleneck, Bottleneck::ThreadLimited);
        assert_eq!(est.max_tgs_per_cu, Some(4)); // 1024 / 256, thread-bound only
    }

    #[test]
    fn low_tg_memory_stays_thread_limited() {
        // 1KiB/TG at TG=1024 easily fits 32 TGs in the memory pool, but the
        // thread dimension only allows 1 TG of this size to begin with —
        // memory must not spuriously become the reported bottleneck.
        let k = Kernel::new("low_shmem");
        let est = estimate_occupancy(&k, 1024, Some(1024));
        assert_eq!(est.bottleneck, Bottleneck::ThreadLimited);
        assert_eq!(est.max_tgs_per_cu, Some(1));
    }

    #[test]
    fn register_heavy_kernel_reduced_occupancy() {
        let mut k = Kernel::new("regheavy");
        // Push 100 const ops → ~150 regs/thread → occupancy ~85%
        for i in 0..100u32 {
            k.body.push_op(Op::Const { value: i as i64 }, ValueId::new(i));
        }

        let est = estimate_occupancy(&k, 256, None);
        // regs_per_thread = 100 * 1.5 = 150, which exceeds 128 → occupancy < 100%
        assert!(est.occupancy_pct < 100.0);
        assert_eq!(est.bottleneck, Bottleneck::RegisterLimited);
    }

    #[test]
    fn threadgroup_size_limits_occupancy() {
        let k = Kernel::new("bigtg");
        // 2048 threads/tg → capped at 1024.
        let est = estimate_occupancy(&k, 2048, None);
        // 1024/2048 = 0.5
        assert!((est.occupancy_pct - 50.0).abs() < 1.0);
    }

    #[test]
    fn best_threadgroup_size_picks_highest() {
        let k = Kernel::new("best");
        let candidates = &[(64, None), (128, None), (256, None), (512, None), (1024, None)];
        let best = best_threadgroup_size(&k, candidates).unwrap();
        // Empty kernel: all threadgroup sizes give 100%, tie breaks to first (64).
        assert_eq!(best.0, 64);
    }

    // --- Regression coverage for the memory-residency fix ---
    //
    // Before this fix, `mem_occ` was computed as `min(32768/mem_used, 1.0)`,
    // which is structurally always >= 1.0 for any legally-compiled kernel
    // (a single TG's allocation can never exceed the 32KB hard cap) — so
    // `Bottleneck::MemoryLimited` could never be reported and
    // `max_tgs_per_cu` was meaningless. These pin the corrected residency-
    // count model: `mem_tgs = floor(32KB / bytes_per_tg)`, combined with the
    // thread dimension via `min(thread_limited_tgs, mem_limited_tgs)`.

    #[test]
    fn high_tg_memory_reports_single_resident_tg() {
        // moe_gather_qmm_expert_mpp-shaped case: ~20KiB/TG (Ws 4KiB +
        // OutScratch 16KiB). 32768 / 20480 = 1 (floor) -> exactly one
        // resident threadgroup per core, and memory is the binding
        // constraint (thread dimension alone would allow 4 @ TG=256).
        let k = Kernel::new("moe_gather_qmm_expert_mpp_like");
        let est = estimate_occupancy(&k, 256, Some(20 * 1024));
        assert_eq!(est.max_tgs_per_cu, Some(1));
        assert_eq!(est.bottleneck, Bottleneck::MemoryLimited);
        assert!(est.occupancy_pct < 30.0, "expected low occupancy, got {}", est.occupancy_pct);
    }

    #[test]
    fn moderate_tg_memory_reports_three_resident_tgs() {
        // mlx.fast-equivalent case: ~9KiB/TG (register-direct accumulator,
        // no OutScratch). 32768 / 9216 = 3 (floor) -> three resident TGs,
        // still memory-bound relative to the thread dimension (8 @ TG=128).
        let k = Kernel::new("register_direct_accumulator_like");
        let est = estimate_occupancy(&k, 128, Some(9 * 1024));
        assert_eq!(est.max_tgs_per_cu, Some(3));
        assert_eq!(est.bottleneck, Bottleneck::MemoryLimited);
    }

    #[test]
    fn low_tg_memory_is_not_memory_limited() {
        // A small, comfortable TG-memory footprint should never become the
        // reported bottleneck or drag max_tgs_per_cu below what the thread
        // dimension alone allows.
        let k = Kernel::new("small_shmem");
        let est = estimate_occupancy(&k, 256, Some(512)); // 32768/512 = 64 TGs worth
        assert_ne!(est.bottleneck, Bottleneck::MemoryLimited);
        assert_eq!(est.max_tgs_per_cu, Some(4)); // thread-bound: 1024/256
    }
}
