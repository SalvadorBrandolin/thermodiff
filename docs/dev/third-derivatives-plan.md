# Plan: third derivatives in thermodiff's Fortran code generation

Status: proposal, nothing implemented. Written against the code merged from
`refactor/perf-arch` (`2894c85`), at the request of the thesis work on
benchmarking the generated kernels.

Evidence labels: **[M]** measured, with its source; **[D]** derived and
checked here (symbolic check, §2.3); **[R]** read in the source, with
`file:line`; **[S]** suggestion; **[A]** assumption a task must confirm
first. The benchmark-campaign sources cited as `bitacora`, `historia-cambios`
and `brechas` are the author's thesis notes, kept outside this repository
(`papers/mine/thermodiff_claude/`, mostly in Spanish).

---

## 0. Summary

1. **yaeos never uses a third-derivative tensor; every third-order
   quantity it needs comes from finite differences** [R]. The only
   consumers are the critical-point solvers. They need the scalar cubic
   form C(w) = Σ_ijl Ar_ijl w_i w_j w_l along one direction and, for an
   analytic Jacobian, the T and V derivatives of the composition Hessian
   plus one matrix contraction Σ_l Ar_ijl v_l. All of it is in the **T,V
   basis**. The primary targets are therefore **contracted outputs**; the
   full N³ tensor is an optional output, for validation.
2. The brief's third-order chain rule (A2) is correct [D]. Mixed T
   derivatives also need the T derivatives of the nodes (§2.3). For
   quadratic mixing rules C(w) is O(N²); the full tensor is O(N³).
3. **Recommended architecture [S]:** treat the contraction direction as a
   scalar variable s with n → n + s·w. Then C(w) = ∂³Ar/∂s³ and
   Σ_l Ar_ijl w_l = ∂(Ar_ij)/∂s. The existing chain-rule and pass machinery
   handles s like T or V: outputs stay scalar, vector or matrix, and the
   main path needs no rank-3 index handling. The rank-3 tensor path (new
   free index, rank-3 layout and passes) comes second and is optional.
4. **Two latent hazards in today's code [R]:**
   `chain._rate_based_value` (`thermodiff/chain.py:963-998`) returns the
   second-order formula for every `deriv ≥ 2`, so a T³ derivative through
   the log-rate path would be silently wrong (H1). `DiffPlz.__init__`
   computes every attribute of `DERIVATIVE_ATTRIBUTES` when
   `derivatives=None` (`thermodiff/diffplz.py:156-166`), so appending
   third-order attributes there would change the default cost (H2).
5. Byte-identity of every existing kernel is achievable: all third-order
   work sits behind new attributes and options that are absent by default.

---

## 1. Thermodynamic need: what yaeos really uses

### 1.1 Where third derivatives enter yaeos [R]

A grep of `yaeos/src` and `app/critical.f90` finds no third-order output.
The `ArModel` interface stops at second order:
`abs_residual_helmholtz(self, n, v, t, Ar, ArV, ArT, ArTV, ArV2, ArT2, Arn,
ArVn, ArTn, Arn2)` (`src/models/residual_helmholtz/ar_models.f90:86-88`),
and the hyperdual type is "exact first- and second-order"
(`src/adiff/hyperdual.f90:3`). Third-order quantities appear only
implicitly, as finite differences:

**Mixture critical points and lines** (`src/equilibria/critical.f90`,
stability-matrix formulation in the Heidemann-Khalil/Michelsen line):

- `lambda1` (`:536-615`) builds M_ij = √(z_i z_j)·∂ln f_i/∂n_j at
  n = z + s·u∘√z and (V, T) (`lnfug_vt(..., dlnfdn=...)` at `:599`, M at
  `:601-605`) and returns the eigenvalue of smallest |λ| (`:607-609`).
  `dlnfdn` is Arn2_ij/RT + δ_ij/n_i, **T,V basis**
  (`ar_models.f90:641-646`).
- The second criticality condition is a **central FD** of λ₁ along s,
  eps = 1e-4 (`critical.f90:663, 683-685`).
- The Jacobian `df_critical` (`:690-741`) is a central FD of the whole
  residual in X = [a, ln V, ln T, ln P], eps = 1e-10, 1e-3 or 1e-6
  (`:719-740`). Its row 2 is an FD of an FD, i.e. fourth order in effect.
- Same pattern for critical endpoints: `F_cep` (`:789-793`) and `df_cep`
  (`:799-841`). Callers: `critical_point` (`:843-984`), `critical_line`
  (`:99-384`) and the PT envelope critical search
  (`boundaries/phase_envelopes_pt.f90:447,497`).

**Pure critical points** (`src/equilibria/pure_critical_solver.f90`): the
residuals are (P_V, P_VV), with P_VV an FD of P_V, `DV_STEP = 2e-4`
(`:33, 260-266`); P_VV = −Ar_VVV − 2nRT/V³ in (T,V). The (V,T) Jacobian
is also an FD (`:273-308`, `DT_STEP` at `:34`).

Phase envelopes, three-phase lines, flashes and `stability.f90` use only
ln φ / ln f and their first derivatives; the grep finds no third-order use
there.

### 1.2 The exact quantities [D]

By Hellmann-Feynman (M symmetric, λ₁ simple, u its unit eigenvector), with
w = u∘√z and RT fixed:

    dλ₁/ds |_{s=0} = Σ_ijl w_i w_j w_l ∂²ln f_i/∂n_j∂n_l
                   = (1/RT) Σ_ijl Ar_ijl w_i w_j w_l − Σ_i w_i³/z_i²

So `F_critical(2)` needs **one scalar per evaluation**, the cubic form
C(w) in T,V; neither the tensor nor a matrix contraction. An analytic
λ₁ row of `df_critical` over (a, ln V, ln T) would need wᵀ(∂_T Arnn)w,
wᵀ(∂_V Arnn)w (mixed third derivatives, contracted twice) and
Σ_l Ar_ijl Δz_l with Δz = z_i − z_0 (plus the a-dependence of the √z
weights). The F₂ row would need fourth order and should stay an FD of the
analytic C(w), which already beats today's FD of an FD.

Consequences:

1. **Basis: T,V only.** yaeos works in (n, V, T) everywhere, and the
   chain-rule path falls back when P is a variable (`chain.py:31-33`).
2. **Primary outputs**, by value: C(w); wᵀAr_Tnn w and wᵀAr_Vnn w (or the
   matrices Ar_Tnn, Ar_Vnn, already O(N²)); Σ_l Ar_ijl v_l; Ar_VVV and
   Ar_TVV for pure critical points.
3. **The full tensor Ar_ijl is not consumed by yaeos** [R]. It serves
   validation (§5) and users outside yaeos.

### 1.3 Proposed workloads (exact output sets) [S]

Each workload names the calculation it serves, and T10.2 checks that it
really serves it. This avoids repeating the W5b error: W5b is
`dT,dV,dT2,dTdV` (`benchmarks/td_bench/hooks.py:124`) and was described as
giving Cp and the speed of sound, which need ArV2.

| Id | Outputs | Serves | Check that it serves it |
|---|---|---|---|
| W7a | ArV, ArV2, ArV3, ArTV, ArTV2 | Pure critical (P_V, P_VV) and analytic (V,T) Jacobian | Replace the FD of `residual_critical_eqs`/`jacobian_critical_eqs` in a copy; same constants to solver tolerance |
| W7b | Arn2, C(w) | F₁, F₂ of `F_critical` without the FD in s | Same λ₁, F₂ as `critical.f90:681-685` to O(eps²); critical line unchanged |
| W7c | W7b + wᵀAr_Tnn w, wᵀAr_Vnn w, Σ_l Ar_ijl v_l, ArVn, ArTn | Analytic λ₁ row of `df_critical` | Column-wise agreement with the FD Jacobian |
| W7d | Ar_ijl (packed or mirrored) | Validation, non-yaeos users | mpmath oracle |
| W7e | Ar_Tnn, Ar_Vnn (matrices) | Matrix form of W7c for other codes | mpmath oracle |

---

## 2. IR, DiffPlz and chain-rule changes

### 2.1 Current state [R]

- `thermodiff/derivatives.py:30-61`: `DERIVATIVE_ATTRIBUTES` (14),
  `FIRST_ORDER`, `SECOND_ORDER`, `_FRIENDLY`, `_ALIASES`, `_REQUIRED`.
  `_DERIV_SUFFIX` and `_DERIV_ORDER` (`codegen/common.py:16-34`) take the
  emission order from `DERIVATIVE_ATTRIBUTES`.
- `DiffPlz._compute` (`diffplz.py:170-219`) builds each attribute from a
  lower one. `diff_ni`/`diff_dnidnj` (`:221-310`) reduce the Kronecker δ
  and merge `Piecewise` conditions for the two free indices i, j.
- `chain.py` knows orders 1 and 2 only: `_ATTR_VARS` (`:55-65`), the
  `_FIRST/_SECOND_VARIABLES/_ATTRIBUTE` maps (`:1301-1329`),
  `_needed_for_attribute` (`:1746`), `_total_first`/`_total_second`
  (`:2025-2056`), `_assemble_attributes` (`:2310-2339`, length 1 or 2).
  `_scalar_chain`/`_propagate` (`:1650-1730`) do forward **second-order**
  propagation; nested nodes stop at `_total_second_derivative` (`:1420`).
- Log rates: `_component_log_rates`/`_pair_log_rates` (`:872-960`) accept
  any `order` and `_array_names(order)` (`:628`) names `X_TTT`, but
  `_rate_based_value` (`:986-998`) implements deriv 1 and "else" = the
  deriv-2 formula (**H1**).
- Indices: `thermovars.py` reserves i, j for derivatives and k, l, m for
  sums; `_order_free` (`codegen/common.py:147`) puts i, j first. A third
  free index needs a new reserved symbol.

### 2.2 Changes [S]

1. **Separate attribute tuple (fixes H2).** Add `THIRD_ORDER` and the
   directional attributes to `derivatives.py` **outside**
   `DERIVATIVE_ATTRIBUTES`, computed only when selected; emitted after the
   existing ones so old declaration/output order never moves. Names (yaeos
   name in parentheses, for `yaeos_model.py`): `dt3`, `dv3`, `dt2dv`,
   `dtdv2` (ArT3, ArV3, ArT2V, ArTV2); `dt2dni`, `dv2dni`, `dtdvdni`;
   `dtdnidnj`, `dvdnidnj` (ArTn2, ArVn2); full tensor `dnidnjdnq` (Arn3,
   new index `q`); directional, with input vector w: `dw3` = C(w),
   `dw2dni` = Σ_jl Ar_ijl w_j w_l, `dwdnidnj` = Σ_l Ar_ijl w_l, `dtdw2`,
   `dvdw2`.
2. **Directional variable s (main path).** In `DiffPlz` (reference path):
   `xreplace` of `n[idx]` by `n[idx] + s*w[idx]`, `sp.diff` in s, then
   s → 0. In `chain.py` (fast path): s is a third T/V-like variable whose
   node derivatives are contractions, S_k,s = Σ_i S_k,i w_i,
   S_k,ss = wᵀH_k w, S_k,sss = Σ K_k,ijl w_i w_j w_l. These are themselves
   summation nodes, so sharing, `sum_alpha` and nesting apply; F does not
   depend on s directly. w becomes an input argument `w(nc)`. [A] T3.1
   checks whether `resolve._scan_atoms` already declares such an
   `IndexedBase` as an input array; otherwise add a per-call
   `direction_args` context field, like `integer_args`.
3. **Full tensor (secondary path).** Reserve `q` in `thermovars.py`;
   extend `_order_free` to (i, j, q), a no-op when q is absent; write
   `DiffPlz.diff_dnidnjdnq` by generalising the `Piecewise` merge of
   `diff_dnidnj` to {i, j, q}; make `NameScheme` detect a user index named
   `q`, as it does for `ic, ia, ib` (`chain.py:224`).
4. **chain.py third order.** Length-3 entries in `_ATTR_VARS` and
   `_needed_for_attribute`, a `_THIRD_VARIABLES` map, `_node_third`,
   `_OuterPartials.third`, `_total_third` (§2.3), routed by
   `_assemble_attributes`. Node third derivatives come from `DiffPlz` on
   the small node bodies, as at order 2. `nested_nodes` and `scalar_chain`
   get third order in later tasks; until then a new `_Rule` in
   `options.py:_RULES` disables them for third-order selections, with the
   usual warning.
5. **Log rates to any order (fixes H1).** With A' = A·h,
   A^(k) = Σ_{r=0}^{k−1} C(k−1, r) A^(r) h^(k−1−r); for pairs
   h = c + g_a + g_b, whose T derivatives are the existing
   `_rate_vector_definitions` vectors `g`, `g_T`, `g_TT`. Before that,
   `_rate_based_value` must raise on `deriv > 2` (byte-neutral guard).

### 2.3 Chain rule at third order [D]

With Ar = F(S_1..S_m, T, V), node derivatives g_k = ∂S_k/∂n,
H_k = ∂²S_k/∂n², K_k = ∂³S_k/∂n³ and sums over repeated node indices:

    Ar_ijl = Σ F_kmp g_k,i g_m,j g_p,l
           + Σ F_km (H_k,ij g_m,l + H_k,il g_m,j + H_k,jl g_m,i)
           + Σ F_k K_k,ijl

This is the brief's formula, and it is correct: checked exactly
(difference 0 in rational arithmetic) against `sp.diff` on a toy model with
N = 3 and four nodes (linear B, quadratic D, Σn, and a cubic, T-dependent
node E). The mixed T form, also checked (difference 0), with S_k,T = ∂_T S_k
and ∂_T(·) the partial of F at fixed nodes:

    Ar_Tij = Σ F_T,k H_k,ij + Σ F_k ∂_T H_k,ij + Σ F_T,km g_k,i g_m,j
           + Σ F_km (S_k,T H_m,ij + ∂_T g_k,i g_m,j + ∂_T g_k,j g_m,i)
           + Σ F_kmp S_k,T g_m,i g_p,j

So mixed attributes need the node derivatives `dt`, `dtdni`, `dtdnidnj`
(and the V analogues), not only the n-derivatives.

**Low rank, quadratic mixing** (PR76/SRK nodes Σn, B = Σn_i b_i,
D = ΣΣ n_i n_j a_ij): H_Σn = H_B = 0, H_D = 2a is constant and every K is
0. Hence Ar_ijl = Σ F_kmp g_k⊗g_m⊗g_p + 3·sym(Σ_m F_Dm H_D⊗g_m). The
cubic form C(w) = Σ F_kmp (g_k·w)(g_m·w)(g_p·w) + 3 Σ_m F_Dm (wᵀH_D w)
(g_m·w) costs m dot products and **one** symmetric product wᵀaw: O(N²),
against O(N³) for the tensor. The matrix contraction
Σ_l Ar_ijl v_l = Σ F_kmp (g_p·v) g_k g_mᵀ + Σ_m F_Dm [(g_m·v) H_D +
(H_D v) g_mᵀ + g_m (H_D v)ᵀ] is O(N²), the cost class of the Hessian. For
CMR (cubic mixing node Σ L_ijl n_i n_j n_l), K = 6L is constant and C(w)
is O(N³), the cost class of Ar itself.

---

## 3. Output layout

| Output | Shape | Layout [S] |
|---|---|---|
| C(w), wᵀAr_Tnn w, wᵀAr_Vnn w, ArV3, ArTV2, … | scalar | as today |
| Σ_jl Ar_ijl w_j w_l, `dt2dni`, … | (nc) | as today |
| Σ_l Ar_ijl v_l, Ar_Tnn, Ar_Vnn | (nc,nc), symmetric | reuse the `hessian_symmetry` mirroring, generalised from `attr == "dnidnj"` (`pipeline.py:1889-1895`) to a set of symmetric rank-2 attributes |
| Ar_ijl | (nc,nc,nc) or packed nc(nc+1)(nc+2)/6 | new option `tensor3_layout`: `full_mirrored` (fill i≤j≤q, copy to the five permutations; default) or `packed` (tetrahedral, index function documented in the kernel header) |

Rank-3 outputs need `render._declare_output`/`_render_output` with three
free indices (`render.py:174-240`; `mirror` is limited to
`len(free_order) == 2` at `:214`), and a registered `_render_definition`
handler if a new record type is added (`ir.py`, `render.py:631-643`). Note
[R]: `sym_rank3` (`options.py:202`, `symmetry.py:36-120`) canonicalises
rank-3 symmetric **parameters** inside sums; it is not output symmetry and
gives no tetrahedral loop.

**Memory.** [M, `reportes/bitacora-generador.md` E8] gfortran puts
automatic arrays on the heap without `-fstack-arrays`; above ≈128 kB of
temporaries glibc returns memory on each `free`, which caused jumps of up
to ≈2× and ≈71 000 page faults per run. At N = 50 the full tensor is
125 000 doubles = 1.0 MB and the packed one 22 100 doubles = 177 kB (still
over the threshold); N² contractions are 20 kB. Hence [S]: rank-3 outputs
are always caller-supplied dummy arguments, never automatic arrays; no
rank-3 temporaries (the low-rank assembly of §4 streams into the output);
benchmark profile HIGHSA primary, O3 and HIGH secondary.

---

## 4. Passes: carry-over, guards and new passes

Passes: `_PASSES` in `codegen/pipeline.py`, table in
`docs/source/architecture.md` §"Passes in execution order". Measured
effects: `reportes/historia-cambios.md`, `bitacora-generador.md`.

**Carry over unchanged** (they act on sums and temporaries, not on output
rank): `share_sum_temps`, `array_cse`, `sum_alpha`/`sum_free_alpha` (E5),
`hoist_*` (E7, E10), `fuse_temps` with `fuse_guard`, `comp_fuse` (it17),
`keep_libm_tables`, `fuse_libm_guard`, `contiguous_args`, `table_layout`,
`drop_pair_tables`, `share_elementwise`, `split_affine` (E21), and
`split_acc=2` only without `-ffast-math` (E22). On the directional path
they see only rank ≤ 2 outputs; [A] T6.1 confirms this pass by pass.

**Need an operation-count guard or a rank gate** [M: unguarded
transformations made things worse]:

- `sep_sums` unguarded took CMR from 3.30 to 6.69 (E14, it05); `"cost"` is
  safe (it08). Keep `cost` and extend its Σ N^depth estimate to N³ nests.
- `dummy_canon` without the sharing criterion: CMR W3 0.87 → 1.77, W6
  0.80 → 1.71 (E18, it21); with it, it only improves. At third order there
  are 3! images per sum: keep the criterion and re-validate it.
- `nest_cost` (it16) nests every sum over ≥ 3 indices. The tensor and
  CMR's C(w) create such sums; check its choice of inner sum against the
  tetrahedral loop.
- `lowrank_outputs` / `_lowrank_bilinear` (`hoisting.py:470`) assume
  `free = (first, second)`: gate them explicitly to rank-2 outputs.
- `scalar_chain="estimate"` (E22: PC-SAFT generation of ten kernels
  649 → 464 s; p22 reference 367 s) is second order only: disable it via
  `_RULES` for third-order selections until T4.4.

**Measured neutral, left out of third-order presets** [M,
`historia-cambios.md:130`]: `col_major`, `cse_expensive`, `float_canon`.

**New passes** [S], each with its own option, off by default, and a guard
comparing `count_ops`/`scalar_operation_count` before and after:

1. `tetra_fold`: symmetric rank-3 outputs and temporaries iterate
   i ≤ j ≤ q (≈N³/6), then mirror or pack; symmetry proved with
   `_canonical_symmetric` over the three free indices.
2. `lowrank_tensor3`: write Ar_ijl as Σ c·sym(x⊗y⊗z) + Σ sym(M⊗y) from
   the chain-rule structure (§2.3); rejected when the per-element cost is
   not lower, as `_lowrank_bilinear` does.
3. `contraction_pushdown`: rewrite Σ_ijl (Σ_k c_k g_k,i g_k,j g_k,l) w_i
   w_j w_l as Σ_k c_k (g_k·w)³. The s-variable design (§2.2) gives most of
   this for free; the pass is a safety net for user-written contractions,
   decided after T3.3 is measured.

---

## 5. Validation

All oracles exist **before** any generator change (P1). [S]

- **mpmath oracle** at 50 digits on the `DiffPlz` expression: exact
  `sp.diff` for N ≤ 4; for every model and N ∈ {2, 3, 8, 20, 50}, a
  univariate Taylor expansion along s (n + s·w, T + s, V + s) with
  `mpmath.taylor`, which gives C(w), the mixed contractions and single
  tensor entries by polarisation without the N³ expansion. Relative
  tolerance 1e-12; a failure is a bug, never fixed by widening tolerance.
- **FD of the second-order kernels**: central differences in n of the
  existing Arn2, ArTn, ArVn kernels (O(h²), h ≈ 1e-4 relative); an
  independent check built on the existing golden kernels.
- **Symmetry**: the six permutations of (i, j, q) agree to the last bit
  for mirrored storage; packed round-trips are exact.
- **Consistency**: C(w) equals Σ Arn3 w w w; Σ_l Arn3_ijl v_l equals
  `dwdnidnj`; the mixed T/V outputs match FD of Arn2 in T and V.
- **Thermodynamic**: on a copy of yaeos, W7b reproduces `F_critical(2)` to
  FD accuracy, and the critical line of `app/critical.f90` (PR78, 12
  components) is unchanged to solver tolerance.

---

## 6. Reference implementations and AD baselines

- **Lean hand-written kernels** for PR76 and SRK (C(w) by §2.3 and
  Σ_l Ar_ijl v_l): the fair reference ("esbelto", `bitacora` E11). Lean
  CMR and PC-SAFT kernels are optional.
- **Tapenade tangent-over-tangent-over-reverse** (`AR_D_D_B`): seeds
  d1 = d2 = w give Σ_jl Ar_ijl w_j w_l in one call, whose dot product with
  w is C(w); T/V seeds give wᵀAr_Tnn w. It extends the `AR_D_B`/`AR_D_D`
  scheme of `reportes/ad-completo.md` §"Semántica de llamadas por modo".
- **One truncated-Taylor type** (the user wants a single type): generalise
  the yaeos `hyperdual` with order mode (`ad-completo.md` §"Hiperduales:
  modo de orden y planificador") to a univariate Taylor type of order ≤ 3
  along one direction. C(w) then costs one evaluation; Σ_l Ar_ijl v_l
  costs N evaluations by polarisation. Caveat [M, same section]:
  `hd_order` is module-global state and does not allow concurrent threads
  with different modes. The new type should carry its order per value or
  per call.
- **AD wrappers request only what the workload needs**: one call per
  directional quantity, never the full tensor contracted afterwards. Call
  counts are checked as in `bench_v2/test/hd_eval_counts.f90`.

---

## 7. Benchmark design

- **Protocol** [M, the requester's campaigns]: dedicated core (CPU 5),
  performance governor, epochs, methods interleaved inside each epoch,
  median of per-epoch ratios against `ref-ia` or the lean kernel; the
  method is resolved to an integer outside the timed region (E15.1: string
  dispatch cost 3.3 ns per comparison and biased small N); no other heavy
  load on the machine.
- **Profiles**: HIGHSA primary; O3 and HIGH secondary; `split_acc` only in
  O3.
- **Grid**: PR76, SRK → RKPR, CMR → PC-SAFT; N ∈ {2, 4, 8, 20, 50};
  W7a-W7e; methods lean, td variants, ad-tap, ad-taylor.
- **Variants**: append `stage(...)` entries to
  `benchmarks/td_bench/registry.py`, one changed setting per stage; add
  W7* to `WORKLOADS` in `hooks.py`.
- **Generation time**: record `diff_time_s` and emit time per workload ×
  variant (`hooks.generate_workload`); set a per-model budget before
  measuring kernels.
- **Wrapper**: a yaeos-side extension type with a new type-bound procedure
  (yaeos's `residual_helmholtz` has no slot). Its `select case` mask
  dispatch mirrors `yaeos_model.py:323-367`; the existing dispatch text
  stays byte-identical. Measure wrapper against direct calls as in
  `envoltorios-yaeos.md` [M there: ≤ 6 % at N = 8, noise at N = 50].

---

## 8. Risks

| Risk | Evidence | Mitigation |
|---|---|---|
| Generation time explodes for PC-SAFT | [M] pcsaft e3 w6 already 57.7 s cold after the refactor (`docs/dev/refactor-perf-arch-report.md:23,33`); p32 w6 62.0 s; ten requester kernels 464 s (`bitacora` E22). F has C(m+4,3) third partials for m nodes plus T, V [D] | Directional path first; measure PR76/SRK generation before PC-SAFT; per-model budget; third-order `scalar_chain` only if needed |
| Piecewise blow-up in `diff_dnidnjdnq` | [R] `diff_dnidnj` already merges conditions pairwise (`diffplz.py:251-310`) | Tensor path is secondary; the chain rule keeps node bodies small |
| Silent wrong T³ derivative | [R] H1 (`chain.py:986-998`) | Guard first (T0.3), then Leibniz (T4.1), oracle-tested |
| Default cost change through `DiffPlz(derivatives=None)` | [R] H2 (`diffplz.py:156-166`) | Separate `THIRD_ORDER` tuple (T2.1) |
| Heap temporaries ≥ 128 kB | [M] E8 | No rank-3 automatic arrays; caller buffers; HIGHSA |
| Code size of full-tensor kernels | [A] | Packed layout; tensor only on request |
| Rank-2 passes meet rank-3 or rank-0 residuals | [R] `_lowrank_bilinear`, `mirror` | Explicit rank gates (T6.2) and tests |
| Namer renumbering breaks the pins | [R] architecture.md §Invariants | New passes inactive with options off; pins rerun in every task |

---

## 9. Task sequence (refactor style)

Rules, as in the refactor: one logical change per commit; G1 before each
commit and G2 before each merge; golden fast, full and complete tiers
byte-identical for **all existing cases at every merge**; one new option or
preset decision per task; every new transformation lands with its guard;
third-order golden cases go to a **new tier** `third`, created only by
T1.4. Sizes: S ≤ ½ day, M 1-3 days, L > 3 days.

| Task | Deliverable | Verified by | Risk | Size |
|---|---|---|---|---|
| **P0** | *Pins and baselines (byte-neutral)* | | | |
| T0.1 | Pins: no third-order path reachable with today's selections; SHA pins of `DERIVATIVE_ATTRIBUTES`, `_DERIV_ORDER`, `_PASSES` order | `tests/codegen/test_pipeline_pins.py` | none | S |
| T0.2 | Generation-time baselines (PR76, SRK, CMR w6) on a quiet machine | `docs/dev/perf_log.md` entry | load noise | S |
| T0.3 | `_rate_based_value` raises on `deriv > 2` (H1 guard) | unit test; golden complete unchanged | none | S |
| **P1** | *Oracles before code* | | | |
| T1.1 | mpmath Taylor oracle: C(w), contractions, tensor entries, mixed T/V | self-test on an analytic toy F | slow at N = 50 | M |
| T1.2 | FD-of-Hessian checker over existing kernels | agrees with T1.1 (PR76, N = 3) | step choice | S |
| T1.3 | Lean PR76/SRK C(w) and Σ_l Ar_ijl v_l in Fortran | T1.1 | none | M |
| T1.4 | Golden tier `third` (empty manifest, tooling in `tools/golden_cases.py`) | `check_golden --tier third` runs | none | S |
| **P2** | *DiffPlz reference path* | | | |
| T2.1 | `THIRD_ORDER` and directional attributes outside `DERIVATIVE_ATTRIBUTES`; names, aliases, `_REQUIRED` | `tests/test_derivatives.py`; golden unchanged | name clashes | S |
| T2.2 | `DiffPlz` T/V third order and mixed one-n attributes | T1.1 | none | S |
| T2.3 | Directional s-attributes in `DiffPlz` (n → n + s·w) | T1.1 | SymPy cost on large models | M |
| T2.4 | Plain (non-chain) `to_fortran` for third-order selections: scalar/vector/matrix outputs, `w` input | compile + T1.1 on PR76/SRK; first `third` golden cases | `_scan_atoms` behaviour | M |
| **P3** | *Chain rule, directional path* | | | |
| T3.1 | Confirm or add direction input-argument handling | signature unit test | — | S |
| T3.2 | `chain.py` length-3 maps, `_node_third`, `_OuterPartials.third`, `_total_third` | T1.1; `sp.diff` equality on toy models (§2.3) | 50-line function limit: split helpers | L |
| T3.3 | s as a node variable (contracted node derivatives as summation nodes) | T1.1; op count vs T2.4 | sharing misses | M |
| T3.4 | `_RULES`: disable `scalar_chain`, `nested_nodes` for third-order selections (reported) | `tests/test_codegen_options.py` | none | S |
| **P4** | *T-derivative machinery* | | | |
| T4.1 | Leibniz log-rate derivatives to any order (removes the H1 guard) | T1.1 with T³; golden unchanged, `comp_fuse` cases included | order-2 path must stay untouched | M |
| T4.2 | Mixed `dtdnidnj`, `dvdnidnj`, `dtdw2`, `dvdw2` by chain rule | T1.1, T1.2 | — | M |
| T4.3 | Nested nodes at third order | T1.1 | — | M |
| T4.4 | Third-order `scalar_chain` propagation, only if F-partials dominate (T0.2, T10.1) | T1.1; generation time | complexity | L |
| **P5** | *Tensor path (optional)* | | | |
| T5.1 | Index `q`, `_order_free`, `NameScheme` clash detection | golden unchanged | — | S |
| T5.2 | `diff_dnidnjdnq` (Piecewise over three indices) | T1.1 for N ≤ 4 | Piecewise blow-up | M |
| T5.3 | Rank-3 declaration and render; option `tensor3_layout` | symmetry tests; T1.1 | — | M |
| **P6** | *Passes* | | | |
| T6.1 | Audit every pass on third-order selections; list rank assumptions | report and tests | — | S |
| T6.2 | Rank gates (`lowrank_outputs`; `mirror` for a set of symmetric rank-2 attributes) | golden unchanged | — | S |
| T6.3 | `tetra_fold` pass with guard | op counts; T1.1 | namer order | M |
| T6.4 | `lowrank_tensor3` pass with guard | op counts; T1.1 | — | M |
| T6.5 | Re-validate the `dummy_canon` criterion for 3! images; `sep_sums=cost` depth for N³ | measured, not assumed | 2× regressions (E18) | M |
| T6.6 | `contraction_pushdown`, only if T3.3 leaves gaps | op counts | — | M |
| **P7** | *Wrappers and AD* | | | |
| T7.1 | `yaeos_model.py`: new procedure and mask for third-order kernels; existing text unchanged | `tests/test_yaeos_model.py`; golden | interface choice (Q5) | M |
| T7.2 | Tapenade `AR_D_D_B` baselines (PR76, SRK, CMR, PC-SAFT) | T1.1 | toolchain | M |
| T7.3 | Single truncated-Taylor type (order ≤ 3, order per value) | T1.1; call counts | yaeos patch scope | L |
| **P8-P10** | *Preset, benchmarks, integration* | | | |
| T8.1 | Preset decision (Q4) | `tests/test_presets.py`; golden | — | S |
| T9.1 | W7a-W7e in `hooks.py`; `stage` entries in `registry.py` | `tests/test_bench_hooks.py` | — | S |
| T10.1 | Benchmark campaign (§7) | protocol | machine load | M |
| T10.2 | yaeos demo: `F_critical(2)` from W7b, analytic λ₁ row from W7c, pure critical from W7a | critical line unchanged; Newton iteration counts | — | M |

Critical path to value in yaeos: T0.x → T1.1 → T2.1-T2.4 → T3.1-T3.4 →
T4.1-T4.2 → T7.1 → T10.2. The tensor path (P5, T6.3-T6.4) can wait.

---

## 10. Open questions for the user

1. **Contracted outputs or the full tensor as the default target?**
   Recommendation: contracted (C(w), wᵀAr_Tnn w, wᵀAr_Vnn w,
   Σ_l Ar_ijl v_l); yaeos consumes nothing else (§1). The full tensor stays
   available on request, for validation.
2. **T,V or T,P basis?** Recommendation: T,V only. The yaeos critical code
   is T,V (`critical.f90:599`, `ar_models.f90:641-646`), and `chain_rule`
   falls back when P is a variable. A T,P cubic form can be assembled from
   T,V outputs through the implicit-function relations if it is ever
   needed.
3. **Which models first?** Recommendation: PR76 and SRK (lean references,
   O(N²) C(w)) to prove correctness and gains; then CMR (the N³ case, which
   tests `nest_cost`/`tetra_fold`); PC-SAFT last, after a generation-time
   budget from T0.2; GE models only if a GE-based mixing rule needs third
   order in composition.
4. **Do third derivatives need a new preset?** Recommendation: not at
   first. `standard` stays unchanged and third-order attributes are opt-in
   by selection. Third-order options enter `fastest` only after T10.1 shows
   they help (one decision, T8.1). A preset only for third order would
   duplicate `fastest` without evidence.
5. **Where does the yaeos entry point live?** `ArModel%residual_helmholtz`
   has no slot. Options: (a) an extension type in the thermodiff wrappers
   with a new type-bound procedure; (b) a yaeos patch adding an optional
   `residual_helmholtz3` to `ArModel`, with a default by FD or Taylor AD.
   Recommendation: (a) now; (b) proposed upstream once T10.2 shows the
   gain.
6. **Are the critical-solver changes (T10.2) part of this project or a
   separate yaeos task?** Recommendation: separate. This plan delivers the
   kernels and a demonstration on a copy.
7. **Are the AD baselines limited to Tapenade and one Taylor type (no
   hyperdual-of-hyperdual)?** Recommendation: yes, as the brief requires.
