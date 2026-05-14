#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

// glass::nvidia::tuning — per-shape SIMT-vs-cuBLASDx dispatch decisions.
//
// The auto-dispatching wrappers in l2.cuh / l3.cuh consult this table BEFORE
// falling back to a static heuristic. Entries are populated by measurement;
// see bench/run_bench.py and bench/autotune.py.
//
// Two tables exist:
//
//   kGlobalTable — shipped with GLASS, community-grown via PRs.
//
//   kLocalTable  — opt-in per-build override. If the user defines
//                  GLASS_TUNING_TABLE_LOCAL to point at a header (e.g.
//                  "bench/tuning/tuning_myhost_sm120.cuh"), that header is
//                  included below and is expected to define the symbol
//                  `kLocalTable` as a `std::array<entry, N>`. Entries here
//                  take precedence over kGlobalTable.
//
// Decision priority inside should_use_cublasdx*<>:
//   1. Local table hit  → use that entry's use_cublasdx flag.
//   2. Global table hit → use that entry's flag.
//   3. Otherwise        → fall back to the API-specific compile-time heuristic.
//
// This header is included inside `namespace glass::nvidia { ... }` (by
// glass-nvidia.cuh), so symbols here live in glass::nvidia::tuning::.

namespace tuning {

    enum class api : uint8_t {
        gemm,
        gemv,
        row_strided_gemm,
        row_strided_gemv,
        gemm_batched_1d,
    };

    // Per-shape measurement entry. BATCH=1 for non-batched APIs.
    // Sizes are uint16 because the largest BlockDim cuBLASDx targets fits;
    // SM is the compute capability times ten (e.g. 860 for sm_8.6).
    struct entry {
        api      op;
        uint16_t M;
        uint16_t N;
        uint16_t K;
        uint16_t BATCH;
        uint16_t SM;
        bool     use_cublasdx;
    };

    // -----------------------------------------------------------------------
    // Shipped global table — grows over time via community contributions.
    // To contribute: run `python3 bench/autotune.py --emit-pr-diff` and open
    // a PR appending your entries here. See bench/TUNING.md.
    // -----------------------------------------------------------------------
    constexpr std::array<entry, 0> kGlobalTable = {};

    // -----------------------------------------------------------------------
    // Per-build local override.
    // The override header must define `kLocalTable` inside this namespace as
    // a `constexpr std::array<entry, N> kLocalTable = {{ ... }};`. See
    // bench/TUNING.md for the format and the autotune.py script for an
    // automated way to generate it.
    // -----------------------------------------------------------------------
#ifdef GLASS_TUNING_TABLE_LOCAL
    #include GLASS_TUNING_TABLE_LOCAL
#else
    constexpr std::array<entry, 0> kLocalTable = {};
#endif

    // Compile-time linear scan. `n` is small in practice (tens of entries per
    // API per SM at most) so a linear scan is fine — and is constexpr.
    constexpr int find_in(const entry* tbl, std::size_t n, api op,
                          uint32_t M, uint32_t N, uint32_t K,
                          uint32_t BATCH, uint32_t SM)
    {
        for (std::size_t i = 0; i < n; ++i) {
            if (tbl[i].op == op &&
                tbl[i].M == M && tbl[i].N == N && tbl[i].K == K &&
                tbl[i].BATCH == BATCH && tbl[i].SM == SM)
                return static_cast<int>(i);
        }
        return -1;
    }

    template <api Op, uint32_t M, uint32_t N, uint32_t K,
              uint32_t BATCH, uint32_t SM>
    constexpr int find_local()
    {
        return find_in(kLocalTable.data(), kLocalTable.size(),
                       Op, M, N, K, BATCH, SM);
    }

    template <api Op, uint32_t M, uint32_t N, uint32_t K,
              uint32_t BATCH, uint32_t SM>
    constexpr int find_global()
    {
        return find_in(kGlobalTable.data(), kGlobalTable.size(),
                       Op, M, N, K, BATCH, SM);
    }

    // Top-level lookup: tristate enum so callers can both get the decision
    // and learn which source supplied it (for print_dispatch_full).
    enum class source : uint8_t { none, local, global, heuristic };

    struct decision {
        bool   use_cublasdx;
        source from;
        int    table_index;  // valid if from == local or global
    };

    template <api Op, uint32_t M, uint32_t N, uint32_t K,
              uint32_t BATCH, uint32_t SM>
    constexpr decision lookup(bool heuristic_says)
    {
        constexpr int li = find_local<Op, M, N, K, BATCH, SM>();
        if constexpr (li >= 0)
            return { kLocalTable[li].use_cublasdx, source::local, li };
        constexpr int gi = find_global<Op, M, N, K, BATCH, SM>();
        if constexpr (gi >= 0)
            return { kGlobalTable[gi].use_cublasdx, source::global, gi };
        return { heuristic_says, source::heuristic, -1 };
    }

} // namespace tuning
