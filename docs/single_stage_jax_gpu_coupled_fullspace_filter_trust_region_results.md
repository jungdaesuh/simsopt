# Coupled Fullspace Filter/Trust-Region Results

**Route:** `CFS-FTR1`

**Disposition:** `CLOSED_BOUNDED_NEGATIVE / NON_PROMOTING`

**Comparative verdict:** `NOT_PRODUCED`

## Authoritative gate

The sole complete RTX 5090 ten-step Gate 2 receipt is:

`/home/jungdaesuh/campaigns/cfs-ftr1-gate2-r2-20260810T0520Z/gates/ftr-canary-10/gate-receipt.json`

- source HEAD: `cf61b250d9535b769a860d0495789fdc29dc0a70`
- gate receipt SHA-256:
  `60f511f0d1cc95cd9584ee86e54587077cd674922562fdd697db676bfd6d57b9`
- raw result SHA-256:
  `583c8c074efb9e16b63c45f635574b7d6c275633ac6d86104d79fefc6b4f44ee`
- GPU-memory SHA-256:
  `0673474b15c9efa7c95375eb2d19d0bad30903f4d413d386706cca517919f4e6`
- runtime-evidence SHA-256:
  `e9ba186db229a5dcdb7cf3130550597349599f2c177d184ae8a3fe40936d5539`

The public validator rehashed and semantically revalidated the immutable
receipt. All four evidence files are mode `0444`.

## Gate result

The gate status is `FAIL` with frozen reasons:

- `MODEL_FILTER_DECISIONS`
- `FEASIBILITY_NOT_MAINTAINED`
- `RAW_KKT_NOT_DECREASED`

The decisive numerical failures are independent of the telemetry-type reason:

| Quantity | Initial | Final | Result |
|---|---:|---:|---|
| physical objective | `8.444212891013206e-05` | `3.7401545317545477e-05` | decreased |
| scaled feasibility infinity norm | `7.802097876149912e-16` | `1.6264351433058275e-05` | failed |
| raw KKT stationarity infinity norm | `5.108879270420846e-03` | `1.7202560952732145e-02` | failed |

The optimizer accepted 7 of 10 attempts, used 18 joint evaluations, and ended
at radius `0.125`. Its normal, tangency, multiplier-projection, KKT, and Schur
linear-solve certificates all passed. The model/filter reason records that the
raw history encoded accepted flags as integers while the frozen receipt gate
required JSON booleans; the two numerical failures already close the route.

## Performance evidence

The synchronized ten-step solve took `9.905568286 s`, giving the frozen
diagnostic projection `99.05568286 s` for 100 iterations versus the
`287.30421751597896 s` engineering threshold. This is not a speed result:
Gate 2 failed convergence progress, so no endpoint solve, warm RTX timing,
A100 run, or native comparison was authorized.

The timed optimizer reported zero hot H2D and D2H transfers. Peak bound GPU
memory was `0.7579967491642898` of physical memory, below the `0.8` ceiling.

## Pre-receipt attempt

`/home/jungdaesuh/campaigns/cfs-ftr1-gate2-20260810T0515Z` is
`NOT_PRODUCED`: strict JSON serialization rejected nonfinite placeholders in
sparse rejected-step history before any raw result or gate receipt was
published. Commit `cf61b250d` added null-safe telemetry serialization and its
regression. This attempt is not a gate verdict.

## Closure

Per the frozen SSOT, no CFS-FTR1 tuning or replay follows a complete failed
Gate 2. The route demonstrates that the device-resident coupled formulation is
fast enough in its ten-step diagnostic, but it does not maintain feasibility or
improve raw KKT stationarity. Therefore it cannot progress to endpoint
certification or comparative timing.
