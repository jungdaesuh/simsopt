---
date: 2026-06-04
artifact_kind: gpd_analysis
subject: design-only proxy current fields reaching topology gates
status: active_pattern
---

# Design-Only Proxy Field Topology Fingerprint

## Pattern

Treat a finite-current proxy field as topology-invalid when all three signals
are present:

1. Boozer/VMEC iota lift ratio >= 5.
2. Midplane poloidal null count > 1.
3. Phi-hit collapse >= 10x versus the matched real-coil or VMEC-current field.

This is an AND fingerprint, not three independent rejection rules. A single
signal can arise from legitimate resolution, seed, or field-model differences;
the three together indicate the prescribed line-current proxy has become a
design-only singular field, not a production topology field.

## Contract

- Use `DESIGN_ONLY_NO_TOPOLOGY_GATE` in `results.json` as the persisted sidecar
  marker for these fields.
- In-process producers also set `_design_only_no_topology_gate` on the live
  `BiotSavart` object, but JSON reloads are governed by the sidecar.
- Poincare and topology gates must reject the field unless an explicit diagnostic
  override is recorded in the produced metrics.

## Evidence Anchor

The proxy validation bundle showed the combined signature: Boozer lift inflated
by more than 5x versus VMEC, multiple midplane nulls, and a collapsed phi-hit
pattern compared with the validated VMEC-current reference. FIX-3 VMEC curtor
is the validated iota-lift oracle for the boundary; filament and bundled
proxy-line fields are design fields only.
