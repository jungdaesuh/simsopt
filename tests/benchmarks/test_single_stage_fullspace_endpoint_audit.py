from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from benchmarks.single_stage_fullspace_endpoint_audit import (
    AUTHORITY_SCHEMA_VERSION,
    PREFLIGHT_SCHEMA_VERSION,
    CertificatePayloadBinding,
    CrossEvaluatorAuthority,
    EndpointAuditAuthority,
    EndpointAuditAuthorityError,
    EndpointAuditDisposition,
    EndpointAuditOutputDisposition,
    EndpointVectorAuthority,
    EvaluatorAuthority,
    ExactBranchAuthority,
    FieldLineAuthority,
    MissingAuthority,
    ProjectionAuthority,
    ProvenanceAuthority,
    preflight_endpoint_audit_authority,
    not_produced_output,
    write_output,
    write_preflight,
)
from benchmarks.single_stage_fullspace_snapshot import (
    ArtifactRef,
    canonical_json_bytes,
    load_canonical_json_bytes,
)
from simsopt_jax.solve.fullspace import FullSpaceRoute


def _write_artifact(
    root: Path,
    relative_path: str,
    schema_version: str,
    body: dict[str, object] | None = None,
) -> ArtifactRef:
    document = {"schema_version": schema_version, **({} if body is None else body)}
    payload = canonical_json_bytes(document)
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return ArtifactRef(
        relative_path=relative_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        schema_version=schema_version,
    )


def _vector_authority(
    root: Path, relative_path: str, offset: float
) -> EndpointVectorAuthority:
    vector = [offset + float(index) for index in range(716)]
    artifact = _write_artifact(
        root,
        relative_path,
        "endpoint-vector-v1",
        {"endpoint": {"dtype": "float64", "physical_state": vector}},
    )
    return EndpointVectorAuthority(
        artifact=artifact,
        vector_path=("endpoint", "physical_state"),
        dtype_path=("endpoint", "dtype"),
        vector_sha256=hashlib.sha256(canonical_json_bytes(vector)).hexdigest(),
    )


def _authority(root: Path) -> EndpointAuditAuthority:
    source = _write_artifact(root, "authority/source.json", "source-manifest-v1")
    runtime = _write_artifact(root, "authority/runtime.json", "runtime-evidence-v1")
    bootstrap = _write_artifact(root, "authority/bootstrap.json", "bootstrap-v1")
    native_configuration = _write_artifact(
        root, "authority/native-evaluator.json", "native-evaluator-v1"
    )
    jax_configuration = _write_artifact(
        root, "authority/jax-evaluator.json", "jax-evaluator-v1"
    )
    branch_configuration = _write_artifact(
        root, "authority/branch.json", "branch-authority-v1"
    )
    fieldline_configuration = _write_artifact(
        root, "authority/fieldline.json", "fieldline-authority-v1"
    )
    return EndpointAuditAuthority(
        schema_version=AUTHORITY_SCHEMA_VERSION,
        route=FullSpaceRoute.CFS_SQP1,
        candidate_endpoint=_vector_authority(root, "endpoints/candidate.json", 0.0),
        historical_native_endpoint=_vector_authority(
            root, "endpoints/native.json", 1.0
        ),
        cross_evaluator=CrossEvaluatorAuthority(
            native=EvaluatorAuthority(
                evaluator="native",
                source_manifest=source,
                runtime_evidence=runtime,
                configuration=native_configuration,
            ),
            jax=EvaluatorAuthority(
                evaluator="jax",
                source_manifest=source,
                runtime_evidence=runtime,
                configuration=jax_configuration,
            ),
        ),
        exact_branch=ExactBranchAuthority(
            configuration=branch_configuration,
            newton_tolerance=1.0e-13,
            maximum_newton_iterations=20,
            reproduced_state_infinity_tolerance=1.0e-10,
            basin_classification_rule="compare continuation labels and root state",
            material_branch_switch_rule="reject changed continuation/root label",
        ),
        projection=ProjectionAuthority(),
        field_line=FieldLineAuthority(
            configuration=fieldline_configuration,
            radial_seeds=(1.0, 1.1, 1.2, 1.3),
            vertical_seeds=(0.0, 0.0, 0.0, 0.0),
            phi_sections=(0.0, 0.25, 0.5, 0.75),
            integrator="simsoptpp.fieldline_tracing-adaptive",
            integration_tolerance=1.0e-16,
            maximum_integration_iterations=20_000,
            turn_count=80,
            closure_tolerance=1.0e-8,
            traced_iota_estimator="full-torus-phi0-return-section-atan2",
        ),
        provenance=ProvenanceAuthority(
            source_manifest=source,
            runtime_evidence=runtime,
            bootstrap_artifact=bootstrap,
        ),
        certificate_payload=CertificatePayloadBinding.current(),
    )


def test_complete_authority_is_ready_but_preflight_is_never_promoting(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path)

    preflight = preflight_endpoint_audit_authority(tmp_path, authority)

    assert preflight.schema_version == PREFLIGHT_SCHEMA_VERSION
    assert preflight.disposition is EndpointAuditDisposition.READY
    assert preflight.missing_authority == ()
    assert preflight.endpoint_work_authorized
    assert not preflight.promotion_eligible
    assert len(preflight.authority_sha256) == 64
    assert len(preflight.certificate_payload_sha256) == 64


def test_missing_historical_endpoint_and_fieldline_are_not_produced(
    tmp_path: Path,
) -> None:
    authority = replace(
        _authority(tmp_path),
        historical_native_endpoint=None,
        field_line=None,
    )

    preflight = preflight_endpoint_audit_authority(tmp_path, authority)

    assert preflight.disposition is EndpointAuditDisposition.NOT_PRODUCED
    assert preflight.missing_authority == (
        MissingAuthority.HISTORICAL_NATIVE_ENDPOINT,
        MissingAuthority.FIELDLINE,
    )
    assert not preflight.endpoint_work_authorized
    assert not preflight.promotion_eligible


@pytest.mark.parametrize(
    "missing",
    (MissingAuthority.HISTORICAL_NATIVE_ENDPOINT, MissingAuthority.FIELDLINE),
)
def test_each_missing_optional_authority_independently_blocks_work(
    tmp_path: Path,
    missing: MissingAuthority,
) -> None:
    authority = _authority(tmp_path)
    if missing is MissingAuthority.HISTORICAL_NATIVE_ENDPOINT:
        authority = replace(authority, historical_native_endpoint=None)
    else:
        authority = replace(authority, field_line=None)

    preflight = preflight_endpoint_audit_authority(tmp_path, authority)

    assert preflight.disposition is EndpointAuditDisposition.NOT_PRODUCED
    assert preflight.missing_authority == (missing,)
    assert not preflight.endpoint_work_authorized


def test_candidate_artifact_tampering_fails_before_authorization(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path)
    candidate_path = tmp_path / authority.candidate_endpoint.artifact.relative_path
    candidate_path.write_bytes(candidate_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="size mismatch"):
        preflight_endpoint_audit_authority(tmp_path, authority)


def test_endpoint_vector_shape_and_digest_are_independently_bound(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path)
    wrong_digest = replace(
        authority.candidate_endpoint,
        vector_sha256="0" * 64,
    )
    with pytest.raises(EndpointAuditAuthorityError, match="vector digest mismatch"):
        preflight_endpoint_audit_authority(
            tmp_path, replace(authority, candidate_endpoint=wrong_digest)
        )

    wrong_size = replace(authority.candidate_endpoint, coordinate_count=715)
    with pytest.raises(EndpointAuditAuthorityError, match="coordinate_count"):
        preflight_endpoint_audit_authority(
            tmp_path, replace(authority, candidate_endpoint=wrong_size)
        )


def test_tolerance_and_certificate_contract_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path)
    with pytest.raises(EndpointAuditAuthorityError, match="projection tolerances"):
        preflight_endpoint_audit_authority(
            tmp_path,
            replace(
                authority,
                projection=replace(
                    authority.projection,
                    objective_tolerance=2.0e-15,
                ),
            ),
        )

    with pytest.raises(EndpointAuditAuthorityError, match="Newton tolerance"):
        preflight_endpoint_audit_authority(
            tmp_path,
            replace(
                authority,
                exact_branch=replace(
                    authority.exact_branch,
                    newton_tolerance=2.0e-13,
                ),
            ),
        )

    stale_binding = replace(
        authority.certificate_payload,
        endpoint_fields=authority.certificate_payload.endpoint_fields[:-1],
    )
    with pytest.raises(EndpointAuditAuthorityError, match="live contract"):
        preflight_endpoint_audit_authority(
            tmp_path,
            replace(authority, certificate_payload=stale_binding),
        )


def test_fieldline_protocol_requires_exact_complete_inputs(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    assert authority.field_line is not None
    invalid_fieldline = replace(authority.field_line, vertical_seeds=(0.0,))

    with pytest.raises(EndpointAuditAuthorityError, match="R/Z seed vectors"):
        preflight_endpoint_audit_authority(
            tmp_path, replace(authority, field_line=invalid_fieldline)
        )


def test_preflight_writer_emits_canonical_nonpromoting_output(
    tmp_path: Path,
) -> None:
    preflight = preflight_endpoint_audit_authority(
        tmp_path,
        replace(
            _authority(tmp_path),
            historical_native_endpoint=None,
            field_line=None,
        ),
    )
    output = tmp_path / "preflight.json"

    write_preflight(output, preflight)

    document = load_canonical_json_bytes(output.read_bytes())
    assert output.read_bytes() == canonical_json_bytes(preflight.to_payload())
    assert isinstance(document, dict)
    assert document["disposition"] == "NOT_PRODUCED"
    assert document["promotion_eligible"] is False


def test_missing_authority_emits_evidence_free_not_produced_output(
    tmp_path: Path,
) -> None:
    preflight = preflight_endpoint_audit_authority(
        tmp_path,
        replace(
            _authority(tmp_path),
            historical_native_endpoint=None,
            field_line=None,
        ),
    )
    output = not_produced_output(preflight)
    output_path = tmp_path / "endpoint-audit.json"

    write_output(output_path, output, tmp_path)

    document = load_canonical_json_bytes(output_path.read_bytes())
    assert output.disposition is EndpointAuditOutputDisposition.NOT_PRODUCED
    assert isinstance(document, dict)
    assert document["promotion_eligible"] is False
    assert document["native_on_jax"] is None
    assert document["jax_on_historical_native"] is None
    assert document["field_line"] is None
