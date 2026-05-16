# Lens 1 — Technical Correctness

**Model:** Opus 4.7 (max effort)
**Scope:** Independently verify every load-bearing technical claim in `.artifacts/lm_minpack_port_plan_2026-05-16/PLAN.md` rev 2 against authoritative sources.

## Verifications performed

### 1. SciPy `least_squares(method='lm')` defaults

**Probe:**
```bash
/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax/bin/python -c "
import inspect, scipy.optimize as so
print('least_squares:', inspect.signature(so.least_squares))
print('leastsq:', inspect.signature(so.leastsq))
"
```

**Result (scipy 1.17.1):**
```
least_squares: ftol=1e-08, xtol=1e-08, gtol=1e-08, ...
leastsq:       ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0, ...
```

**Verdict:** Plan claim correct in rev 2 (post-first-critic-pass): `least_squares` uses `1e-8` uniformly; legacy `leastsq` uses `1.49012e-8`. ✓

### 2. MINPACK Fortran line references

Files downloaded from netlib.org and verified:

| Citation | Plan claim | Actual file content | Verdict |
|---|---|---|---|
| `lmder.f:313` | `if (gnorm .le. gtol) info = 4` | `if (gnorm .le. gtol) info = 4` | ✓ |
| `lmder.f:297-306` | `gnorm` computation | `gnorm = zero; ...; gnorm = dmax1(gnorm,dabs(sum/wa2(l)))` | ✓ |
| `lmder.f:312` | (verification question, not direct cite) | comment line `c` — info=2 set at line 420, info=3 at 421-422 | ✓ (plan doesn't directly cite 312) |
| `lmder.f:421-422` | info=3 = conjunction info=1 ∧ info=2 | `if (... .and. info .eq. 2) info = 3` | ✓ |
| `lmder.f:431` | (rev 3 cite) info=8 | `if (gnorm .le. epsmch) info = 8` | ✓ |
| `lmder_serial.f:381-396` | par escalation logic | **HTTP 404 — `lmder_serial.f` does not exist on netlib** | ✗ **FINDING L1F1** |
| `lmder_serial.f:418-432` | info codes 0-8 | **HTTP 404** | ✗ **FINDING L1F1 (same)** |
| `lmpar.f:222` | 10-iteration cap | line 222 is `* .and. temp .lt. zero .or. iter .eq. 10) go to 220` — iter==10 cap is here, fused with primary exit | ✓ |
| `lmpar.f:151-187` | parl/paru bracket setup | parl computation (forward triangular solve) + paru via `gnorm = enorm(n,wa1); paru = gnorm/delta` | ✓ |
| `lmpar.f:229-230` | secondary-exit predicate | **Lines 229-230 are `180 continue` and `do 210 j = 1, n` — loop continuations, NOT exit predicate.** Actual secondary-exit predicate is on lines 220-222 (fused into compound `if`). | ✗ **FINDING L1F2** |
| `qrfac.f:109` | strict `>` pivot tie-break | `if (rdiag(k) .gt. rdiag(kmax)) kmax = k` | ✓ |

### 3. JAX QR bit-equality on (384,40) — empirical reproduction

**Probe:**
```bash
/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax/bin/python -c "
import os; os.environ['JAX_ENABLE_X64']='True'
import jax; jax.config.update('jax_enable_x64', True)
import numpy as np
import jax.scipy.linalg as jsl
import scipy.linalg as sla
rng = np.random.default_rng(0)
A = rng.standard_normal((384, 40)).astype(np.float64)
Qsp, Rsp, Psp = sla.qr(A, mode='economic', pivoting=True)
Qjx, Rjx, Pjx = jsl.qr(A, mode='economic', pivoting=True)
print('P eq:', np.array_equal(Psp, np.asarray(Pjx)))
print('R bit-eq:', np.array_equal(Rsp, np.asarray(Rjx)))
print('Q bit-eq:', np.array_equal(Qsp, np.asarray(Qjx)))
print('Q max diff:', float(np.max(np.abs(Qsp - np.asarray(Qjx)))))
"
```

**Result:**
```
P eq:        True
R bit-eq:    True
Q bit-eq:    False
Q max diff:  4.163336342344337e-17
```

**Verdict:** Plan claim "Q max diff = 4.163e-17 on (384,40)" reproduced exactly. ✓

**Additional probe:** `dir(jax.lax.linalg)` shows `householder_product`, `householder_product_p`, `ormqr`, `ormqr_p`, `qr`, `qr_p` — **no `geqrf`** as the plan claims. ✓

### 4. `optimizer_jax.py` line references

All cited line refs verified at HEAD via `grep -n`:

| Cite | Plan claim | Actual | Verdict |
|---|---|---|---|
| :1340 | `_gmres_solve_least_squares_system` | `def _gmres_solve_least_squares_system(` | ✓ |
| :1416 | `_lm_iteration` | `def _lm_iteration(flat_residual_fn, state, *, tol):` | ✓ |
| :1403 | `_lm_defaults` | `def _lm_defaults(dtype):` | ✓ |
| :1522 | `levenberg_marquardt` (host) | `def levenberg_marquardt(` | ✓ |
| :1636 | `levenberg_marquardt_traceable` | `def levenberg_marquardt_traceable(` | ✓ |
| :1510-1511 | LM `success` line | `"success": finite_candidate & (... <= tol)` | ✓ |
| :1403-1410 | damping factors | `expand_factor=4.0, shrink_factor=0.5, mild_shrink_factor=0.8, ratio_low=0.25, ratio_high=0.75` | ✓ |
| :1467-1490 | asymmetric trust-region `lax.cond` | matches | ✓ |
| :1492-1512 | result schema | matches | ✓ |
| :155 | VALID_LEAST_SQUARES_ALGORITHMS | `frozenset({"quasi-newton", "lm"})` | ✓ |
| :634 | `resolve_target_least_squares_optimizer_method` | matches | ✓ |

### 5. Environment

```
jax 0.10.0  jaxlib 0.10.0  scipy 1.17.1  numpy 2.4.3
```
✓ Matches plan claim.

### 6. JAX PR #25955

```
$ gh pr view 25955 --repo jax-ml/jax --json title,mergedAt,state
{
  "mergedAt": "2025-02-12T18:05:10Z",
  "state": "MERGED",
  "title": "Enable pivoted QR on GPU via MAGMA"
}
```

Plan claim "Feb 2025" matches; the PR is real and merged. Note: the PR's stated purpose is GPU-via-MAGMA — CPU `dgeqp3` FFI was added earlier. The plan's framing of #25955 as "the PR that added pivoted QR" is accurate as the GPU enabler but slightly imprecise as a CPU-FFI attribution. Not flagged as a finding because the plan's surrounding text ("dispatches to the same `lapack_<t>geqp3_ffi` symbol SciPy uses on CPU and to `cu_hybrid_geqp3` (MAGMA) on GPU") is internally correct.

### 7. `scipy.linalg.qr_multiply`

```
$ python -c "from scipy.linalg import qr_multiply; import inspect; print(inspect.signature(qr_multiply))"
(a, c, mode='right', pivoting=False, conjugate=False, overwrite_a=False, overwrite_c=False)
```

Plan's `mode='left', pivoting=True` call signature is valid. ✓

## Findings

```
FINDING L1F1: lmder_serial.f cited at PLAN.md:144, 151 doesn't exist on netlib (HTTP 404)
Location: PLAN.md:144 (par escalation cite), PLAN.md:151 (info codes cite)
Lens: technical-correctness
Confidence: 98
Severity: major (citation correctness — load-bearing for any reader trying to verify against source)
Evidence:
  - `curl -m 10 -o /dev/null -w 'HTTP: %{http_code}\n' 'https://www.netlib.org/minpack/lmder_serial.f'` → `HTTP: 404`
  - Full netlib /minpack/ directory listing has: chkder.f, dogleg.f, dpmpar.f, enorm.f, fdjac1.f, fdjac2.f, hybrd.f, hybrd1.f, hybrj.f, hybrj1.f, lmder.f, lmder1.f, lmdif.f, lmdif1.f, lmpar.f, lmstr.f, lmstr1.f, qform.f, qrfac.f, qrsolv.f, rwupdt.f, r1mpyq.f, r1updt.f. No `lmder_serial.f`.
  - The cited content IS at the cited line numbers in `lmder.f` (no `_serial` suffix):
    - lmder.f:381-396 = par escalation block (verified)
    - lmder.f:418-432 = info codes 1, 2, 3, 5, 6, 7, 8 cascade (verified)
Fix: rename `lmder_serial.f:` → `lmder.f:` at both occurrences (line numbers stay identical).
```

```
FINDING L1F2: lmpar.f:229-230 cited as secondary-exit predicate location; actual lines 229-230 are loop continuations
Location: PLAN.md:252 ("(lmpar.f:229-230)")
Lens: technical-correctness
Confidence: 95
Severity: major (citation correctness)
Evidence (verbatim from netlib lmpar.f):
   220:          if (dabs(fp) .le. p1*delta
   221:      *       .or. parl .eq. zero .and. fp .le. temp
   222:      *            .and. temp .lt. zero .or. iter .eq. 10) go to 220
   ...
   229:   180       continue
   230:          do 210 j = 1, n
   ...
   255:   220 continue   ! actual exit target
The plan's stated secondary-exit predicate `parl == 0 AND fp ≤ prev_fp AND prev_fp < 0` matches exactly the second clause of the line 220-222 compound `if` (where `temp` = `prev_fp` per line 213 `temp = fp`). Not lines 229-230.
Fix: change `lmpar.f:229-230` to `lmpar.f:220-222`; note the secondary-exit is fused into the primary-exit / 10-iter cap compound `if`.
```

## Verified-correct (no finding)

Probed and matches plan claim:
- SciPy `least_squares` defaults `ftol=xtol=gtol=1e-8` (scipy 1.17.1)
- SciPy `leastsq` legacy defaults `ftol=xtol=1.49012e-8, gtol=0.0`
- JAX 0.10.0 / jaxlib 0.10.0 / scipy 1.17.1 / numpy 2.4.3 runtime versions
- JAX QR bit-equality on (384,40): R/P bit-equal, Q max diff = 4.163e-17 (exact reproduction)
- `jax.lax.linalg` exposes `qr`, `householder_product`, `ormqr` but no `geqrf` (correctly stated by plan)
- `scipy.linalg.qr_multiply(a, c, mode='left', pivoting=True)` is a valid call signature
- JAX PR #25955 merged 2025-02-12, title "Enable pivoted QR on GPU via MAGMA"
- `lmder.f:313` `if (gnorm .le. gtol) info = 4`
- `lmder.f:297-306` gnorm computation
- `lmder.f:421-422` info=3 conjunction (matrix-free)
- `lmpar.f:222` 10-iteration cap (fused with secondary-exit)
- `lmpar.f:151-187` parl + paru bracket setup
- `qrfac.f:109` strict `>` pivot tie-break (`if (rdiag(k) .gt. rdiag(kmax))`)
- All 11 `optimizer_jax.py` line references at HEAD (`:1340`, `:1416`, `:1403`, `:1522`, `:1636`, `:1510-1511`, `:1403-1410`, `:1467-1490`, `:1492-1512`, `:155`, `:634`)

## Files used as oracle (downloads in /tmp during review)

- `/tmp/lmder.f` (netlib MINPACK)
- `/tmp/lmpar.f` (netlib MINPACK)
- `/tmp/qrfac.f` (netlib MINPACK)
- `/tmp/fl_minpack.f90` (fortran-lang/minpack/src/minpack.f90)

## Lens summary

Two concrete citation errors found (L1F1, L1F2). All other 30+ load-bearing technical claims probed verify against authoritative sources. The plan's empirical claims (Q drift on (384,40), JAX API surface, SciPy defaults) all hold under independent reproduction.
