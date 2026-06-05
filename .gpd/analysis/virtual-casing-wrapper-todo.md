---
date: 2026-06-04
artifact_kind: gpd_todo
subject: virtual-casing finite-current topology wrapper
status: deferred
---

# Virtual-Casing Wrapper TODO

P1 #5 is intentionally deferred to a separate phase.

Until that phase lands, topology claims traced on finite-current proxy fields
must use the design-only enforcement path:

- `DESIGN_ONLY_NO_TOPOLOGY_GATE` in the artifact sidecar.
- `POINCARE_ALLOW_DESIGN_ONLY_FIELD` only for explicit diagnostic rendering.
- `design_only_override` recorded in Poincare metrics whenever the diagnostic
  override is used.

The future wrapper should convert a validated current source into the topology
field used for claims, without routing prescribed proxy line-current fields
through production topology gates.
