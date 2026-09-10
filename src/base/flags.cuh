#pragma once
#include <cstdint>

// ─── Shared compile-time dispatch flags ──────────────────────────────────────
//
// Named enums for the triangular / symmetric op families (BLAS-style UPLO/DIAG).
// Included (inside `namespace glass`) by trsv.cuh, trsm.cuh, and syrk.cuh, so
// the same `FillMode`/`Diag` spell every triangle/diagonal choice — no raw
// positional `bool` soup at call sites. `TRANSPOSE` stays a plain `bool`
// template flag across the library (matching gemm's `TRANSPOSE_A/B`).

/** @brief Which triangle of a symmetric/triangular matrix is stored/touched (BLAS UPLO). */
enum class FillMode : uint32_t { Lower = 0, Upper = 1, Full = 2 };

/** @brief Whether the diagonal is implicitly unit (not read) or explicit (BLAS DIAG). */
enum class Diag : uint32_t { NonUnit = 0, Unit = 1 };
