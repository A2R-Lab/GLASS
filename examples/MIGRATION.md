# GLASS BLAS-convention migration cheatsheet

GLASS now follows the standard BLAS / cuBLAS / NumPy / Eigen conventions. Most
call sites change mechanically. The one subtle case is the `gemm` template-arg
swap: it is **silent when your matrices are square**, so audit non-square calls.

Column-major is the default everywhere (Fortran / Eigen default).

## 1. `gemm` — contraction moved to the last dim (`<M,N,K>` → `<M,K,N>`)

```cpp
// C is now M×N with contraction K (was: middle dim was the contraction).
//   op(A) is M×K (TRANSPOSE_A ⇒ A is K×M);  op(B) is K×N (TRANSPOSE_B ⇒ B is N×K).

- glass::gemm<float, M, N, K>(alpha, A, B, beta, C);   // OLD: A M×N, B N×K, C M×K
+ glass::gemm<float, M, K, N>(alpha, A, B, beta, C);   // NEW: C M×K, contraction N → swap last two args

// runtime form: same swap
- glass::gemm<float>(m, n, k, alpha, A, B, beta, C);
+ glass::gemm<float>(m, k, n, alpha, A, B, beta, C);
```

New capability (genuinely rectangular, no squareness assumption):

```cpp
glass::gemm<float, M, N, K, /*TA=*/true,  /*TB=*/false>(alpha, A, B, beta, C); // C = Aᵀ·B
glass::gemm<float, M, N, K, /*TA=*/false, /*TB=*/true >(alpha, A, B, beta, C); // C = A·Bᵀ
glass::gemm<float, M, N, K, /*TA=*/true,  /*TB=*/true >(alpha, A, B, beta, C); // C = Aᵀ·Bᵀ
```

## 2. `gemm_ex` and per-operand `ROW_MAJOR_A` / `ROW_MAJOR_B` are GONE

A row-major operand is a transposed column-major operand — express it with the
transpose flags instead (see `11_rowmajor_is_transpose.cu`):

```cpp
- glass::gemm_ex<float, /*TB*/false, /*RM_A*/true, /*RM_B*/false, /*RM_C*/true>(m,n,k, alpha,A,B,beta,C);
+ glass::gemm<float, M, N, K, /*TA=*/true, /*TB=*/false, /*ROW_MAJOR_C=*/true>(alpha, A, B, beta, C);
//        row-major A (M×K) == col-major K×M ⇒ TRANSPOSE_A;  ROW_MAJOR_C kept as the one output-layout flag.
```

## 3. `gemm_strided` / `gemv_strided` — alpha/beta to the FRONT

```cpp
- glass::gemm_strided<float, M, N, K, A_RS, B_RS>(A, B, C, alpha, beta);
+ glass::gemm_strided<float, M, K, N, A_RS, B_RS>(alpha, A, B, beta, C);  // + the gemm dim swap

- glass::gemv_strided<float, M, N, RS>(A, x, y, alpha, beta);
+ glass::gemv_strided<float, M, N, RS>(alpha, A, x, beta, y);
```

## 4. `l2norm` → `nrm2` (BLAS name; Eigen `x.norm()`)

```cpp
- glass::high_speed::l2norm<float, N>(x, scratch);   glass::warp::l2norm<float>(n, x);
+ glass::high_speed::nrm2  <float, N>(x, scratch);   glass::warp::nrm2  <float>(n, x);
```

## 5. `gemv` — `gemv_ex` removed; `gemv` keeps `ROW_MAJOR`

`glass::gemv<float, TRANSPOSE, ROW_MAJOR>(m, n, alpha, A, x, beta, y)` already
matches BLAS (`y = alpha·op(A)·x + beta·y`); use `TRANSPOSE=true` for `Aᵀx`.

The redundant `gemv_ex` (just `gemv` with the defaults stripped) is GONE — call
`gemv` with explicit flags instead:
```cpp
- glass::gemv_ex<float, /*TRANSPOSE*/false, /*ROW_MAJOR_A*/true>(m, n, alpha, A, x, beta, y);
+ glass::gemv    <float, /*TRANSPOSE*/false, /*ROW_MAJOR  */true>(m, n, alpha, A, x, beta, y);
```
Unlike `gemm`, `gemv` **keeps** its per-matrix `ROW_MAJOR` flag: `TRANSPOSE`
already selects the math op (`A·x` vs `Aᵀ·x`), so a row-major operand cannot also
be expressed as a transpose.

---

Eigen equivalences (column-major): `gemm` = `C.noalias() = alpha*(opA*opB) +
beta*C;`, `nrm2` = `x.norm()`, `asum` = `x.lpNorm<1>()`. The runnable worked
examples for all of the above are `10_gemm_basics.cu`,
`11_rowmajor_is_transpose.cu`, `12_nrm2.cu`, `13_gemm_strided.cu`.
