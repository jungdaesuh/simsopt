# Hardware-First DESC Joint Schema

The JSON config is loaded by
`banana_opt.desc_joint_hardware_spec.load_desc_joint_hardware_spec`. Markdown is
documentation only and is never parsed as configuration.

## JSON Shape

```json
{
  "schema_version": "desc_joint_hardware_spec_v1",
  "hardware_sources": {
    "glb": "/absolute/path/to/live_hardware.glb",
    "hardware_keepout_json": "/absolute/path/to/hardware_keepout.json",
    "hardware_sdf": "/absolute/path/to/hardware_sdf.json",
    "final_oracle": "/absolute/path/to/hardware_contact_report"
  },
  "coil_group_policy": {
    "tf": "fixed",
    "banana": "optimized",
    "proxy": "fixed",
    "vf": "fixed"
  },
  "constraint_names": [
    "coil_length",
    "coil_length_min",
    "coil_coil_spacing",
    "coil_surface_spacing",
    "max_curvature",
    "banana_current",
    "tf_current",
    "width_min",
    "width_max",
    "hardware_keepout"
  ]
}
```

At least one of `hardware_keepout_json` or `hardware_sdf` is required. When both
are present, both are freshness-checked against the live GLB.

## Threshold SSOT

Thresholds are not duplicated in this schema. They resolve through
`banana_opt/hardware_constraint_schema.py`, which imports the hardware constants
from `banana_opt/hardware_contracts.py`.

| Contract | Hardware schema name | Source |
| --- | --- | --- |
| maximum coil length | `coil_length` | `COIL_LENGTH_HARD_LIMIT_M` |
| minimum coil length | `coil_length_min` | `COIL_LENGTH_MIN_TARGET_M` |
| minimum coil-coil spacing | `coil_coil_spacing` | `COIL_COIL_MIN_DIST_M` |
| minimum coil-plasma spacing | `coil_surface_spacing` | `COIL_PLASMA_MIN_DIST_M` |
| maximum curvature | `max_curvature` | `MAX_CURVATURE_INV_M` |
| banana current cap | `banana_current` | `BANANA_CURRENT_HARD_LIMIT_A` |
| TF current cap | `tf_current` | `TF_CURRENT_HARD_LIMIT_A` |
| banana width lower bound | `width_min` | `BANANA_WIDTH_MIN_M` |
| banana width upper bound | `width_max` | `BANANA_WIDTH_MAX_M` |
| hardware keepout | `hardware_keepout` | keepout/SDF provenance plus zero hinge threshold |

The result field names also come from
`hardware_constraint_artifact_payload_field_names`, including
`HARDWARE_CONSTRAINTS_OK` and `HARDWARE_CONSTRAINT_VIOLATIONS`.

## Provenance Gates

The loader fails closed when:

- the GLB path is missing;
- neither keepout JSON nor SDF manifest is provided;
- any referenced path is missing or is a directory;
- a keepout JSON records a different `provenance.glb_sha256` than the live GLB;
- an SDF manifest records a different `provenance.glb_sha256` or stale
  `data_sha256`;
- required hardware constraints are removed from `constraint_names`;
- TF/proxy/VF/banana policy is missing or unknown.

This keeps DESC in-loop steering separate from final promotion. Even when DESC
uses a keepout or SDF objective, promotion still requires a direct SIMSOPT/CAD
hardware oracle result on the loaded exported artifact.

Passed final-oracle evidence is content-bound. The evidence JSON must record:

- `schema_version: "desc_joint_final_oracle_evidence_v1"`;
- `source: "direct_loaded_artifact_hardware_contact_oracle"`;
- `passed: true`;
- `exported_artifact_paths`, matching the validation manifest exactly;
- `exported_artifact_checksums`, matching live SHA-256s of those exported files;
- `source_artifact_checksums`, matching the conversion metadata.

The promotion validator rejects empty exported artifact lists, missing exported
files, stale exported checksums, and source-checksum mismatches.
