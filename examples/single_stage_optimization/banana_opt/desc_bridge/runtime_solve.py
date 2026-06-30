"""Runtime DESC optimization calls for DESC joint banana lanes."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from banana_opt.desc_bridge.runtime_imports import activate_desc_source_root

DescRuntimeSolveStatus = Literal["passed", "failed"]
DescOptimizerNestedOptionValue = int | float | bool
DescOptimizerOptionValue = (
    int | float | bool | dict[str, DescOptimizerNestedOptionValue]
)
_MAX_CONSTRAINT_FEASIBILITY_ROWS = 10


@dataclass(frozen=True, slots=True)
class _DescSaveRequest:
    saveable: object
    final_path: Path
    object_label: str


@dataclass(frozen=True, slots=True)
class DescOptimizerControls:
    """Typed pass-through controls for DESC Optimizer.optimize."""

    ftol: float | None = None
    xtol: float | None = None
    gtol: float | None = None
    ctol: float | None = None
    max_nfev: int | None = None
    max_dx: float | None = None
    initial_trust_radius: float | None = None
    max_trust_radius: float | None = None
    min_trust_radius: float | None = None
    proximal_perturb_order: int | None = None
    proximal_solve_maxiter: int | None = None
    proximal_solve_during_build: bool | None = None

    def tolerances_json_dict(self) -> dict[str, float | None]:
        return {
            "ftol": self.ftol,
            "xtol": self.xtol,
            "gtol": self.gtol,
            "ctol": self.ctol,
        }

    def options_dict(self) -> dict[str, DescOptimizerOptionValue]:
        options: dict[str, DescOptimizerOptionValue] = {}
        if self.max_nfev is not None:
            options["max_nfev"] = self.max_nfev
        if self.max_dx is not None:
            options["max_dx"] = self.max_dx
        if self.initial_trust_radius is not None:
            options["initial_trust_radius"] = self.initial_trust_radius
        if self.max_trust_radius is not None:
            options["max_trust_radius"] = self.max_trust_radius
        if self.min_trust_radius is not None:
            options["min_trust_radius"] = self.min_trust_radius
        perturb_options: dict[str, DescOptimizerNestedOptionValue] = {}
        if self.proximal_perturb_order is not None:
            perturb_options["order"] = self.proximal_perturb_order
        if perturb_options:
            options["perturb_options"] = perturb_options
        solve_options: dict[str, DescOptimizerNestedOptionValue] = {}
        if self.proximal_solve_maxiter is not None:
            solve_options["maxiter"] = self.proximal_solve_maxiter
        if self.proximal_solve_during_build is not None:
            solve_options["solve_during_proximal_build"] = (
                self.proximal_solve_during_build
            )
        if solve_options:
            options["solve_options"] = solve_options
        return options

    def to_json_dict(self) -> dict[str, object]:
        return {
            "tolerances": self.tolerances_json_dict(),
            "options": self.options_dict(),
        }

    def has_proximal_options(self) -> bool:
        return (
            self.proximal_perturb_order is not None
            or self.proximal_solve_maxiter is not None
            or self.proximal_solve_during_build is not None
        )


@dataclass(frozen=True, slots=True)
class DescFixedPolishRuntimeSolveReport:
    status: DescRuntimeSolveStatus
    reason: str
    desc_source_root: Path | None
    desc_version: str | None
    optimizer_method: str
    maxiter: int | None
    verbose: int
    optimizer_controls: DescOptimizerControls
    allow_high_memory_optimizer: bool
    objective_function_type: str | None
    constraint_types: tuple[str, ...]
    input_coilset_type: str | None
    optimized_coilset_type: str | None
    optimized_coilset_path: Path | None
    failed_optimizer_coilset_checkpoint_path: Path | None
    optimizer_success: bool | None
    optimizer_status: int | None
    optimizer_message: str | None
    optimizer_nit: int | None
    optimizer_nfev: int | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "desc_fixed_polish_runtime_solve_report_v1",
            "status": self.status,
            "reason": self.reason,
            "desc_source_root": (
                None if self.desc_source_root is None else os.fspath(self.desc_source_root)
            ),
            "desc_version": self.desc_version,
            "optimizer_method": self.optimizer_method,
            "maxiter": self.maxiter,
            "verbose": self.verbose,
            "optimizer_controls": self.optimizer_controls.to_json_dict(),
            "allow_high_memory_optimizer": self.allow_high_memory_optimizer,
            "objective_function_type": self.objective_function_type,
            "constraint_types": list(self.constraint_types),
            "input_coilset_type": self.input_coilset_type,
            "optimized_coilset_type": self.optimized_coilset_type,
            "optimized_coilset_path": (
                None
                if self.optimized_coilset_path is None
                else os.fspath(self.optimized_coilset_path)
            ),
            "failed_optimizer_coilset_checkpoint_path": (
                None
                if self.failed_optimizer_coilset_checkpoint_path is None
                else os.fspath(self.failed_optimizer_coilset_checkpoint_path)
            ),
            "optimizer_success": self.optimizer_success,
            "optimizer_status": self.optimizer_status,
            "optimizer_message": self.optimizer_message,
            "optimizer_nit": self.optimizer_nit,
            "optimizer_nfev": self.optimizer_nfev,
        }


@dataclass(frozen=True, slots=True)
class DescFixedPolishRuntimeSolveResult:
    optimized_coilset: object
    optimizer_result: object
    report: DescFixedPolishRuntimeSolveReport


@dataclass(frozen=True, slots=True)
class DescJointRuntimeSolveReport:
    status: DescRuntimeSolveStatus
    reason: str
    desc_source_root: Path | None
    desc_version: str | None
    optimizer_method: str
    maxiter: int | None
    verbose: int
    optimizer_controls: DescOptimizerControls
    allow_high_memory_optimizer: bool
    objective_function_type: str | None
    constraint_types: tuple[str, ...]
    input_equilibrium_type: str | None
    input_coilset_type: str | None
    optimized_equilibrium_type: str | None
    optimized_coilset_type: str | None
    optimized_equilibrium_path: Path | None
    optimized_coilset_path: Path | None
    failed_optimizer_equilibrium_checkpoint_path: Path | None
    failed_optimizer_coilset_checkpoint_path: Path | None
    constraint_feasibility_report_path: Path | None
    optimizer_success: bool | None
    optimizer_status: int | None
    optimizer_message: str | None
    optimizer_nit: int | None
    optimizer_nfev: int | None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "desc_joint_runtime_solve_report_v1",
            "status": self.status,
            "reason": self.reason,
            "desc_source_root": (
                None if self.desc_source_root is None else os.fspath(self.desc_source_root)
            ),
            "desc_version": self.desc_version,
            "optimizer_method": self.optimizer_method,
            "maxiter": self.maxiter,
            "verbose": self.verbose,
            "optimizer_controls": self.optimizer_controls.to_json_dict(),
            "allow_high_memory_optimizer": self.allow_high_memory_optimizer,
            "objective_function_type": self.objective_function_type,
            "constraint_types": list(self.constraint_types),
            "input_equilibrium_type": self.input_equilibrium_type,
            "input_coilset_type": self.input_coilset_type,
            "optimized_equilibrium_type": self.optimized_equilibrium_type,
            "optimized_coilset_type": self.optimized_coilset_type,
            "optimized_equilibrium_path": (
                None
                if self.optimized_equilibrium_path is None
                else os.fspath(self.optimized_equilibrium_path)
            ),
            "optimized_coilset_path": (
                None
                if self.optimized_coilset_path is None
                else os.fspath(self.optimized_coilset_path)
            ),
            "failed_optimizer_equilibrium_checkpoint_path": (
                None
                if self.failed_optimizer_equilibrium_checkpoint_path is None
                else os.fspath(self.failed_optimizer_equilibrium_checkpoint_path)
            ),
            "failed_optimizer_coilset_checkpoint_path": (
                None
                if self.failed_optimizer_coilset_checkpoint_path is None
                else os.fspath(self.failed_optimizer_coilset_checkpoint_path)
            ),
            "constraint_feasibility_report_path": (
                None
                if self.constraint_feasibility_report_path is None
                else os.fspath(self.constraint_feasibility_report_path)
            ),
            "optimizer_success": self.optimizer_success,
            "optimizer_status": self.optimizer_status,
            "optimizer_message": self.optimizer_message,
            "optimizer_nit": self.optimizer_nit,
            "optimizer_nfev": self.optimizer_nfev,
        }


@dataclass(frozen=True, slots=True)
class DescJointRuntimeSolveResult:
    optimized_equilibrium: object
    optimized_coilset: object
    optimizer_result: object
    report: DescJointRuntimeSolveReport


class DescFixedPolishRuntimeSolveError(RuntimeError):
    def __init__(self, report: DescFixedPolishRuntimeSolveReport) -> None:
        super().__init__(report.reason)
        self.report = report


class DescJointRuntimeSolveError(RuntimeError):
    def __init__(self, report: DescJointRuntimeSolveReport) -> None:
        super().__init__(report.reason)
        self.report = report


def desc_fixed_equilibrium_polish_setup_failure_report(
    *,
    reason: str,
    desc_source_root: Path | None = None,
    optimizer_method: str = "lsq-exact",
    maxiter: int | None = None,
    verbose: int = 1,
    optimizer_controls: DescOptimizerControls | None = None,
    allow_high_memory_optimizer: bool = False,
) -> DescFixedPolishRuntimeSolveReport:
    """Build the fixed-polish report for failures before the optimizer call."""

    _validate_optimizer_method(optimizer_method)
    _validate_optional_positive_int(maxiter, field_name="maxiter")
    _validate_nonnegative_int(verbose, field_name="verbose")
    optimizer_controls = _coerce_optimizer_controls(optimizer_controls)
    _validate_bool(
        allow_high_memory_optimizer,
        field_name="allow_high_memory_optimizer",
    )
    if not isinstance(reason, str) or reason == "":
        raise ValueError("DESC fixed-polish setup failure reason must be nonempty.")
    return DescFixedPolishRuntimeSolveReport(
        status="failed",
        reason=reason,
        desc_source_root=None if desc_source_root is None else desc_source_root.resolve(),
        desc_version=None,
        optimizer_method=optimizer_method,
        maxiter=maxiter,
        verbose=verbose,
        optimizer_controls=optimizer_controls,
        allow_high_memory_optimizer=allow_high_memory_optimizer,
        objective_function_type=None,
        constraint_types=(),
        input_coilset_type=None,
        optimized_coilset_type=None,
        optimized_coilset_path=None,
        failed_optimizer_coilset_checkpoint_path=None,
        optimizer_success=None,
        optimizer_status=None,
        optimizer_message=None,
        optimizer_nit=None,
        optimizer_nfev=None,
    )


def desc_joint_optimization_setup_failure_report(
    *,
    reason: str,
    desc_source_root: Path | None = None,
    optimizer_method: str = "lsq-exact",
    maxiter: int | None = None,
    verbose: int = 1,
    optimizer_controls: DescOptimizerControls | None = None,
    allow_high_memory_optimizer: bool = False,
) -> DescJointRuntimeSolveReport:
    """Build the joint-mode report for failures before the optimizer call."""

    _validate_optimizer_method(optimizer_method)
    _validate_optional_positive_int(maxiter, field_name="maxiter")
    _validate_nonnegative_int(verbose, field_name="verbose")
    optimizer_controls = _coerce_optimizer_controls(optimizer_controls)
    _validate_bool(
        allow_high_memory_optimizer,
        field_name="allow_high_memory_optimizer",
    )
    if not isinstance(reason, str) or reason == "":
        raise ValueError("DESC joint optimization setup failure reason must be nonempty.")
    return DescJointRuntimeSolveReport(
        status="failed",
        reason=reason,
        desc_source_root=None if desc_source_root is None else desc_source_root.resolve(),
        desc_version=None,
        optimizer_method=optimizer_method,
        maxiter=maxiter,
        verbose=verbose,
        optimizer_controls=optimizer_controls,
        allow_high_memory_optimizer=allow_high_memory_optimizer,
        objective_function_type=None,
        constraint_types=(),
        input_equilibrium_type=None,
        input_coilset_type=None,
        optimized_equilibrium_type=None,
        optimized_coilset_type=None,
        optimized_equilibrium_path=None,
        optimized_coilset_path=None,
        failed_optimizer_equilibrium_checkpoint_path=None,
        failed_optimizer_coilset_checkpoint_path=None,
        constraint_feasibility_report_path=None,
        optimizer_success=None,
        optimizer_status=None,
        optimizer_message=None,
        optimizer_nit=None,
        optimizer_nfev=None,
    )


def run_desc_fixed_equilibrium_polish_runtime(
    *,
    coilset: object,
    objective_function: object,
    constraints: tuple[object, ...] = (),
    output_root: Path,
    desc_source_root: Path | None = None,
    optimizer_method: str = "lsq-exact",
    maxiter: int | None = None,
    verbose: int = 1,
    optimizer_controls: DescOptimizerControls | None = None,
    allow_high_memory_optimizer: bool = False,
) -> DescFixedPolishRuntimeSolveResult:
    """Run guarded fixed-equilibrium coil polish and save only on success."""

    _validate_optimizer_method(optimizer_method)
    _validate_optional_positive_int(maxiter, field_name="maxiter")
    _validate_nonnegative_int(verbose, field_name="verbose")
    optimizer_controls = _coerce_optimizer_controls(optimizer_controls)
    _validate_bool(
        allow_high_memory_optimizer,
        field_name="allow_high_memory_optimizer",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    optimized_coilset_path = output_root / "desc_coils.h5"
    failed_optimizer_coilset_checkpoint_path = (
        output_root / "desc_failed_optimizer_coils.h5"
    )
    desc_version: str | None = None
    optimizer_result: object | None = None
    optimized_coilset: object | None = None
    if not allow_high_memory_optimizer:
        report = _runtime_solve_report(
            status="failed",
            reason=desc_high_memory_optimizer_blocked_reason(
                "fixed-equilibrium polish"
            ),
            desc_source_root=desc_source_root,
            desc_version=None,
            optimizer_method=optimizer_method,
            maxiter=maxiter,
            verbose=verbose,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=allow_high_memory_optimizer,
            objective_function=objective_function,
            constraints=constraints,
            input_coilset=coilset,
            optimized_coilset=None,
            optimized_coilset_path=None,
            failed_optimizer_coilset_checkpoint_path=None,
            optimizer_result=None,
        )
        raise DescFixedPolishRuntimeSolveError(report)
    try:
        with activate_desc_source_root(desc_source_root):
            import desc
            from desc.optimize import Optimizer

            desc_version = _desc_version(desc)
            optimizer = Optimizer(optimizer_method)
            _raise_if_optimizer_rejects_proximal_options(
                optimizer_method=optimizer_method,
                optimizer_class=Optimizer,
                optimizer_controls=optimizer_controls,
            )
            optimized_things, optimizer_result = optimizer.optimize(
                things=coilset,
                objective=objective_function,
                constraints=constraints,
                ftol=optimizer_controls.ftol,
                xtol=optimizer_controls.xtol,
                gtol=optimizer_controls.gtol,
                ctol=optimizer_controls.ctol,
                maxiter=maxiter,
                options=optimizer_controls.options_dict(),
                verbose=verbose,
                copy=True,
            )
            optimized_coilset = _single_optimized_thing(optimized_things)
            optimizer_success = _optimizer_success(optimizer_result)
            if optimizer_success is not True:
                _save_desc_coilset(
                    optimized_coilset,
                    failed_optimizer_coilset_checkpoint_path,
                )
                report = _runtime_solve_report(
                    status="failed",
                    reason=(
                        "DESC fixed-equilibrium polish optimizer did not report "
                        "success=True."
                    ),
                    desc_source_root=desc_source_root,
                    desc_version=desc_version,
                    optimizer_method=optimizer_method,
                    maxiter=maxiter,
                    verbose=verbose,
                    optimizer_controls=optimizer_controls,
                    allow_high_memory_optimizer=allow_high_memory_optimizer,
                    objective_function=objective_function,
                    constraints=constraints,
                    input_coilset=coilset,
                    optimized_coilset=optimized_coilset,
                    optimized_coilset_path=None,
                    failed_optimizer_coilset_checkpoint_path=(
                        failed_optimizer_coilset_checkpoint_path
                    ),
                    optimizer_result=optimizer_result,
                )
                raise DescFixedPolishRuntimeSolveError(report)
            _save_desc_coilset(optimized_coilset, optimized_coilset_path)
            report = _runtime_solve_report(
                status="passed",
                reason="DESC fixed-equilibrium polish optimizer passed.",
                desc_source_root=desc_source_root,
                desc_version=desc_version,
                optimizer_method=optimizer_method,
                maxiter=maxiter,
                verbose=verbose,
                optimizer_controls=optimizer_controls,
                allow_high_memory_optimizer=allow_high_memory_optimizer,
                objective_function=objective_function,
                constraints=constraints,
                input_coilset=coilset,
                optimized_coilset=optimized_coilset,
                optimized_coilset_path=optimized_coilset_path,
                failed_optimizer_coilset_checkpoint_path=None,
                optimizer_result=optimizer_result,
            )
            return DescFixedPolishRuntimeSolveResult(
                optimized_coilset=optimized_coilset,
                optimizer_result=optimizer_result,
                report=report,
            )
    except DescFixedPolishRuntimeSolveError:
        raise
    except Exception as exc:
        report = _runtime_solve_report(
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            desc_source_root=desc_source_root,
            desc_version=desc_version,
            optimizer_method=optimizer_method,
            maxiter=maxiter,
            verbose=verbose,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=allow_high_memory_optimizer,
            objective_function=objective_function,
            constraints=constraints,
            input_coilset=coilset,
            optimized_coilset=optimized_coilset,
            optimized_coilset_path=None,
            failed_optimizer_coilset_checkpoint_path=None,
            optimizer_result=optimizer_result,
        )
        raise DescFixedPolishRuntimeSolveError(report) from exc


def run_desc_joint_optimization_runtime(
    *,
    equilibrium: object,
    coilset: object,
    objective_function: object,
    constraints: tuple[object, ...] = (),
    output_root: Path,
    desc_source_root: Path | None = None,
    optimizer_method: str = "lsq-exact",
    maxiter: int | None = None,
    verbose: int = 1,
    optimizer_controls: DescOptimizerControls | None = None,
    allow_high_memory_optimizer: bool = False,
) -> DescJointRuntimeSolveResult:
    """Run guarded joint equilibrium+coil optimization and save only on success."""

    _validate_optimizer_method(optimizer_method)
    _validate_optional_positive_int(maxiter, field_name="maxiter")
    _validate_nonnegative_int(verbose, field_name="verbose")
    optimizer_controls = _coerce_optimizer_controls(optimizer_controls)
    _validate_bool(
        allow_high_memory_optimizer,
        field_name="allow_high_memory_optimizer",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    optimized_equilibrium_path = output_root / "desc_equilibrium.h5"
    optimized_coilset_path = output_root / "desc_coils.h5"
    failed_optimizer_equilibrium_checkpoint_path = (
        output_root / "desc_failed_optimizer_equilibrium.h5"
    )
    failed_optimizer_coilset_checkpoint_path = (
        output_root / "desc_failed_optimizer_coils.h5"
    )
    desc_version: str | None = None
    optimizer_result: object | None = None
    optimized_equilibrium: object | None = None
    optimized_coilset: object | None = None
    constraint_feasibility_report_path: Path | None = None
    if not allow_high_memory_optimizer:
        report = _joint_runtime_solve_report(
            status="failed",
            reason=desc_high_memory_optimizer_blocked_reason(
                "joint equilibrium+coil"
            ),
            desc_source_root=desc_source_root,
            desc_version=None,
            optimizer_method=optimizer_method,
            maxiter=maxiter,
            verbose=verbose,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=allow_high_memory_optimizer,
            objective_function=objective_function,
            constraints=constraints,
            input_equilibrium=equilibrium,
            input_coilset=coilset,
            optimized_equilibrium=None,
            optimized_coilset=None,
            optimized_equilibrium_path=None,
            optimized_coilset_path=None,
            failed_optimizer_equilibrium_checkpoint_path=None,
            failed_optimizer_coilset_checkpoint_path=None,
            constraint_feasibility_report_path=None,
            optimizer_result=None,
        )
        raise DescJointRuntimeSolveError(report)
    try:
        with activate_desc_source_root(desc_source_root):
            import desc
            from desc.optimize import Optimizer, optimizers

            desc_version = _desc_version(desc)
            optimizer = Optimizer(optimizer_method)
            _raise_if_optimizer_rejects_proximal_options(
                optimizer_method=optimizer_method,
                optimizer_class=Optimizer,
                optimizer_controls=optimizer_controls,
            )
            _raise_if_optimizer_rejects_joint_constraints(
                optimizer_method=optimizer_method,
                optimizer_registry=optimizers,
                optimizer_class=Optimizer,
                constraints=constraints,
                desc_source_root=desc_source_root,
                desc_version=desc_version,
                maxiter=maxiter,
                verbose=verbose,
                optimizer_controls=optimizer_controls,
                allow_high_memory_optimizer=allow_high_memory_optimizer,
                objective_function=objective_function,
                input_equilibrium=equilibrium,
                input_coilset=coilset,
            )
            constraint_feasibility_report_path = (
                _write_joint_constraint_feasibility_report(
                    path=output_root / "desc_constraint_feasibility_report.json",
                    objective_function=objective_function,
                    constraints=constraints,
                )
            )
            optimizer_things = _joint_optimizer_things(
                objective_function=objective_function,
                constraints=constraints,
                equilibrium=equilibrium,
                coilset=coilset,
            )
            optimized_things, optimizer_result = optimizer.optimize(
                things=optimizer_things,
                objective=objective_function,
                constraints=constraints,
                ftol=optimizer_controls.ftol,
                xtol=optimizer_controls.xtol,
                gtol=optimizer_controls.gtol,
                ctol=optimizer_controls.ctol,
                maxiter=maxiter,
                options=optimizer_controls.options_dict(),
                verbose=verbose,
                copy=True,
            )
            optimized_equilibrium, optimized_coilset = _joint_optimized_things(
                optimized_things,
                input_equilibrium=equilibrium,
                input_coilset=coilset,
            )
            optimizer_success = _optimizer_success(optimizer_result)
            if optimizer_success is not True:
                _save_desc_equilibrium_and_coilset(
                    optimized_equilibrium=optimized_equilibrium,
                    equilibrium_path=failed_optimizer_equilibrium_checkpoint_path,
                    optimized_coilset=optimized_coilset,
                    coilset_path=failed_optimizer_coilset_checkpoint_path,
                )
                report = _joint_runtime_solve_report(
                    status="failed",
                    reason=(
                        "DESC joint equilibrium+coil optimizer did not report "
                        "success=True."
                    ),
                    desc_source_root=desc_source_root,
                    desc_version=desc_version,
                    optimizer_method=optimizer_method,
                    maxiter=maxiter,
                    verbose=verbose,
                    optimizer_controls=optimizer_controls,
                    allow_high_memory_optimizer=allow_high_memory_optimizer,
                    objective_function=objective_function,
                    constraints=constraints,
                    input_equilibrium=equilibrium,
                    input_coilset=coilset,
                    optimized_equilibrium=optimized_equilibrium,
                    optimized_coilset=optimized_coilset,
                    optimized_equilibrium_path=None,
                    optimized_coilset_path=None,
                    failed_optimizer_equilibrium_checkpoint_path=(
                        failed_optimizer_equilibrium_checkpoint_path
                    ),
                    failed_optimizer_coilset_checkpoint_path=(
                        failed_optimizer_coilset_checkpoint_path
                    ),
                    constraint_feasibility_report_path=(
                        constraint_feasibility_report_path
                    ),
                    optimizer_result=optimizer_result,
                )
                raise DescJointRuntimeSolveError(report)
            _save_desc_equilibrium_and_coilset(
                optimized_equilibrium=optimized_equilibrium,
                equilibrium_path=optimized_equilibrium_path,
                optimized_coilset=optimized_coilset,
                coilset_path=optimized_coilset_path,
            )
            report = _joint_runtime_solve_report(
                status="passed",
                reason="DESC joint equilibrium+coil optimizer passed.",
                desc_source_root=desc_source_root,
                desc_version=desc_version,
                optimizer_method=optimizer_method,
                maxiter=maxiter,
                verbose=verbose,
                optimizer_controls=optimizer_controls,
                allow_high_memory_optimizer=allow_high_memory_optimizer,
                objective_function=objective_function,
                constraints=constraints,
                input_equilibrium=equilibrium,
                input_coilset=coilset,
                optimized_equilibrium=optimized_equilibrium,
                optimized_coilset=optimized_coilset,
                optimized_equilibrium_path=optimized_equilibrium_path,
                optimized_coilset_path=optimized_coilset_path,
                failed_optimizer_equilibrium_checkpoint_path=None,
                failed_optimizer_coilset_checkpoint_path=None,
                constraint_feasibility_report_path=constraint_feasibility_report_path,
                optimizer_result=optimizer_result,
            )
            return DescJointRuntimeSolveResult(
                optimized_equilibrium=optimized_equilibrium,
                optimized_coilset=optimized_coilset,
                optimizer_result=optimizer_result,
                report=report,
            )
    except DescJointRuntimeSolveError:
        raise
    except Exception as exc:
        report = _joint_runtime_solve_report(
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            desc_source_root=desc_source_root,
            desc_version=desc_version,
            optimizer_method=optimizer_method,
            maxiter=maxiter,
            verbose=verbose,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=allow_high_memory_optimizer,
            objective_function=objective_function,
            constraints=constraints,
            input_equilibrium=equilibrium,
            input_coilset=coilset,
            optimized_equilibrium=optimized_equilibrium,
            optimized_coilset=optimized_coilset,
            optimized_equilibrium_path=None,
            optimized_coilset_path=None,
            failed_optimizer_equilibrium_checkpoint_path=None,
            failed_optimizer_coilset_checkpoint_path=None,
            constraint_feasibility_report_path=constraint_feasibility_report_path,
            optimizer_result=optimizer_result,
        )
        raise DescJointRuntimeSolveError(report) from exc


def _runtime_solve_report(
    *,
    status: DescRuntimeSolveStatus,
    reason: str,
    desc_source_root: Path | None,
    desc_version: str | None,
    optimizer_method: str,
    maxiter: int | None,
    verbose: int,
    optimizer_controls: DescOptimizerControls,
    allow_high_memory_optimizer: bool,
    objective_function: object,
    constraints: tuple[object, ...],
    input_coilset: object,
    optimized_coilset: object | None,
    optimized_coilset_path: Path | None,
    failed_optimizer_coilset_checkpoint_path: Path | None,
    optimizer_result: object | None,
) -> DescFixedPolishRuntimeSolveReport:
    return DescFixedPolishRuntimeSolveReport(
        status=status,
        reason=reason,
        desc_source_root=None if desc_source_root is None else desc_source_root.resolve(),
        desc_version=desc_version,
        optimizer_method=optimizer_method,
        maxiter=maxiter,
        verbose=verbose,
        optimizer_controls=optimizer_controls,
        allow_high_memory_optimizer=allow_high_memory_optimizer,
        objective_function_type=_qualified_type_name(objective_function),
        constraint_types=tuple(
            _qualified_type_name(constraint) for constraint in constraints
        ),
        input_coilset_type=_qualified_type_name(input_coilset),
        optimized_coilset_type=(
            None if optimized_coilset is None else _qualified_type_name(optimized_coilset)
        ),
        optimized_coilset_path=(
            None
            if optimized_coilset_path is None
            else optimized_coilset_path.resolve()
        ),
        failed_optimizer_coilset_checkpoint_path=(
            None
            if failed_optimizer_coilset_checkpoint_path is None
            else failed_optimizer_coilset_checkpoint_path.resolve()
        ),
        optimizer_success=_optimizer_success(optimizer_result),
        optimizer_status=_optimizer_int(optimizer_result, "status"),
        optimizer_message=_optimizer_message(optimizer_result),
        optimizer_nit=_optimizer_int(optimizer_result, "nit"),
        optimizer_nfev=_optimizer_int(optimizer_result, "nfev"),
    )


def _joint_runtime_solve_report(
    *,
    status: DescRuntimeSolveStatus,
    reason: str,
    desc_source_root: Path | None,
    desc_version: str | None,
    optimizer_method: str,
    maxiter: int | None,
    verbose: int,
    optimizer_controls: DescOptimizerControls,
    allow_high_memory_optimizer: bool,
    objective_function: object,
    constraints: tuple[object, ...],
    input_equilibrium: object,
    input_coilset: object,
    optimized_equilibrium: object | None,
    optimized_coilset: object | None,
    optimized_equilibrium_path: Path | None,
    optimized_coilset_path: Path | None,
    failed_optimizer_equilibrium_checkpoint_path: Path | None,
    failed_optimizer_coilset_checkpoint_path: Path | None,
    constraint_feasibility_report_path: Path | None,
    optimizer_result: object | None,
) -> DescJointRuntimeSolveReport:
    return DescJointRuntimeSolveReport(
        status=status,
        reason=reason,
        desc_source_root=None if desc_source_root is None else desc_source_root.resolve(),
        desc_version=desc_version,
        optimizer_method=optimizer_method,
        maxiter=maxiter,
        verbose=verbose,
        optimizer_controls=optimizer_controls,
        allow_high_memory_optimizer=allow_high_memory_optimizer,
        objective_function_type=_qualified_type_name(objective_function),
        constraint_types=tuple(
            _qualified_type_name(constraint) for constraint in constraints
        ),
        input_equilibrium_type=_qualified_type_name(input_equilibrium),
        input_coilset_type=_qualified_type_name(input_coilset),
        optimized_equilibrium_type=(
            None
            if optimized_equilibrium is None
            else _qualified_type_name(optimized_equilibrium)
        ),
        optimized_coilset_type=(
            None if optimized_coilset is None else _qualified_type_name(optimized_coilset)
        ),
        optimized_equilibrium_path=(
            None
            if optimized_equilibrium_path is None
            else optimized_equilibrium_path.resolve()
        ),
        optimized_coilset_path=(
            None
            if optimized_coilset_path is None
            else optimized_coilset_path.resolve()
        ),
        failed_optimizer_equilibrium_checkpoint_path=(
            None
            if failed_optimizer_equilibrium_checkpoint_path is None
            else failed_optimizer_equilibrium_checkpoint_path.resolve()
        ),
        failed_optimizer_coilset_checkpoint_path=(
            None
            if failed_optimizer_coilset_checkpoint_path is None
            else failed_optimizer_coilset_checkpoint_path.resolve()
        ),
        constraint_feasibility_report_path=(
            None
            if constraint_feasibility_report_path is None
            else constraint_feasibility_report_path.resolve()
        ),
        optimizer_success=_optimizer_success(optimizer_result),
        optimizer_status=_optimizer_int(optimizer_result, "status"),
        optimizer_message=_optimizer_message(optimizer_result),
        optimizer_nit=_optimizer_int(optimizer_result, "nit"),
        optimizer_nfev=_optimizer_int(optimizer_result, "nfev"),
    )


def _write_joint_constraint_feasibility_report(
    *,
    path: Path,
    objective_function: object,
    constraints: tuple[object, ...],
) -> Path:
    report = _joint_constraint_feasibility_report(
        objective_function=objective_function,
        constraints=constraints,
    )
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _joint_constraint_feasibility_report(
    *,
    objective_function: object,
    constraints: tuple[object, ...],
) -> dict[str, object]:
    """Summarize joint-optimizer x0 feasibility without combined derivatives."""

    start = time.perf_counter()
    state_report = _desc_objective_state_vector_report(objective_function)
    term_reports = tuple(
        _constraint_term_feasibility_report(constraint) for constraint in constraints
    )
    combined_constraint_report = _combined_constraint_vector_report(constraints)
    evaluated_terms = tuple(
        term_report
        for term_report in term_reports
        if term_report["status"] == "passed"
    )
    failed_terms = tuple(
        term_report
        for term_report in term_reports
        if term_report["status"] == "failed"
    )
    all_finite = bool(
        state_report["state_vector_all_finite"]
        and all(term_report["scaled_value_all_finite"] for term_report in evaluated_terms)
        and all(term_report["scaled_error_all_finite"] for term_report in evaluated_terms)
        and not failed_terms
    )
    constraints_satisfied = bool(
        all_finite
        and all(
            _int_from_report(term_report["violation_count"]) == 0
            for term_report in evaluated_terms
        )
    )
    max_violation = _max_optional_float(
        term_report["max_abs_scaled_error"] for term_report in evaluated_terms
    )
    violation_count = sum(
        _int_from_report(term_report["violation_count"])
        for term_report in evaluated_terms
    )
    return {
        "schema_version": "desc_joint_constraint_feasibility_report_v1",
        "status": "passed" if all_finite else "failed",
        "reason": (
            "DESC joint constraint feasibility diagnostic completed with finite values."
            if all_finite
            else (
                "DESC joint constraint feasibility diagnostic found non-finite "
                "values or unevaluable constraint terms."
            )
        ),
        "objective_function_type": _qualified_type_name(objective_function),
        "constraint_count": len(constraints),
        "state_vector": state_report,
        "constraint_terms": list(term_reports),
        "constraint_terms_all_finite": all_finite,
        "initial_constraints_satisfied": constraints_satisfied,
        "combined_constraint_vector": combined_constraint_report,
        "violation_count": int(violation_count),
        "max_abs_scaled_error": max_violation,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _desc_objective_state_vector_report(objective_function: object) -> dict[str, object]:
    try:
        if not bool(getattr(objective_function, "built", False)) and hasattr(
            objective_function,
            "build",
        ):
            getattr(objective_function, "build")(use_jit=False, verbose=0)
        things = tuple(getattr(objective_function, "things"))
        state_vector = np.asarray(objective_function.x(*things), dtype=float).reshape(-1)
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "thing_count": 0,
            "dim_x": 0,
            "state_vector_all_finite": False,
            "state_vector_finite_count": 0,
            "state_vector_nonfinite_count": 0,
            "state_vector_min": None,
            "state_vector_max": None,
        }
    return {
        "status": "passed" if bool(np.all(np.isfinite(state_vector))) else "failed",
        "reason": "state vector evaluated",
        "thing_count": len(things),
        "dim_x": int(state_vector.size),
        "state_vector_all_finite": bool(np.all(np.isfinite(state_vector))),
        "state_vector_finite_count": _finite_count(state_vector),
        "state_vector_nonfinite_count": _nonfinite_count(state_vector),
        "state_vector_min": _finite_min(state_vector),
        "state_vector_max": _finite_max(state_vector),
    }


def _combined_constraint_vector_report(
    constraints: tuple[object, ...],
) -> dict[str, object]:
    if len(constraints) == 0:
        return {
            "status": "passed",
            "reason": "no DESC hard constraints assembled",
            "dim_x": 0,
            "dim_f": 0,
            "equality_count": 0,
            "inequality_count": 0,
            "scaled_value_all_finite": True,
            "scaled_value_nonfinite_count": 0,
            "slack_vector_all_finite": True,
            "slack_vector_nonfinite_count": 0,
            "slack_vector_in_bounds": True,
            "slack_vector_out_of_bounds_count": 0,
            "invalid_bound_count": 0,
        }
    try:
        from desc.objectives import ObjectiveFunction

        combined_constraints = ObjectiveFunction(
            constraints,
            use_jit=False,
            deriv_mode="blocked",
        )
        combined_constraints.build(use_jit=False, verbose=0)
        things = tuple(getattr(combined_constraints, "things"))
        state_vector = np.asarray(
            combined_constraints.x(*things),
            dtype=float,
        ).reshape(-1)
        scaled_value = np.asarray(
            combined_constraints.compute_scaled(state_vector),
            dtype=float,
        ).reshape(-1)
        lower_bound, upper_bound = tuple(
            np.asarray(bound, dtype=float).reshape(-1)
            for bound in combined_constraints.bounds_scaled
        )
        if lower_bound.shape != scaled_value.shape:
            lower_bound = np.broadcast_to(lower_bound, scaled_value.shape)
        if upper_bound.shape != scaled_value.shape:
            upper_bound = np.broadcast_to(upper_bound, scaled_value.shape)
        equality_mask = lower_bound == upper_bound
        inequality_mask = np.logical_not(equality_mask)
        slack_values = np.clip(
            scaled_value[inequality_mask],
            lower_bound[inequality_mask],
            upper_bound[inequality_mask],
        )
        slack_lower_bound = lower_bound[inequality_mask]
        slack_upper_bound = upper_bound[inequality_mask]
        slack_in_bounds = np.logical_and(
            slack_values >= slack_lower_bound,
            slack_values <= slack_upper_bound,
        )
        invalid_bounds = lower_bound > upper_bound
        return {
            "status": "passed",
            "reason": "combined DESC constraint vector evaluated",
            "dim_x": int(state_vector.size),
            "dim_f": int(scaled_value.size),
            "equality_count": int(np.count_nonzero(equality_mask)),
            "inequality_count": int(np.count_nonzero(inequality_mask)),
            "scaled_value_all_finite": bool(np.all(np.isfinite(scaled_value))),
            "scaled_value_nonfinite_count": _nonfinite_count(scaled_value),
            "slack_vector_all_finite": bool(np.all(np.isfinite(slack_values))),
            "slack_vector_nonfinite_count": _nonfinite_count(slack_values),
            "slack_vector_in_bounds": bool(np.all(slack_in_bounds)),
            "slack_vector_out_of_bounds_count": int(
                np.count_nonzero(np.logical_not(slack_in_bounds))
            ),
            "invalid_bound_count": int(np.count_nonzero(invalid_bounds)),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "dim_x": 0,
            "dim_f": 0,
            "equality_count": 0,
            "inequality_count": 0,
            "scaled_value_all_finite": False,
            "scaled_value_nonfinite_count": 0,
            "slack_vector_all_finite": False,
            "slack_vector_nonfinite_count": 0,
            "slack_vector_in_bounds": False,
            "slack_vector_out_of_bounds_count": 0,
            "invalid_bound_count": 0,
        }


def _constraint_term_feasibility_report(constraint: object) -> dict[str, object]:
    start = time.perf_counter()
    try:
        _build_constraint_term_if_needed(constraint)
        args = constraint.xs(*tuple(getattr(constraint, "things")))
        scaled_value = np.asarray(constraint.compute_scaled(*args), dtype=float).reshape(
            -1
        )
        scaled_error = np.asarray(
            constraint.compute_scaled_error(*args),
            dtype=float,
        ).reshape(-1)
        scaled_lower_bound, scaled_upper_bound = _constraint_scaled_bounds(
            constraint,
            dim_f=int(scaled_value.size),
        )
        value_all_finite = bool(np.all(np.isfinite(scaled_value)))
        error_all_finite = bool(np.all(np.isfinite(scaled_error)))
        violation = np.abs(scaled_error)
        finite_violation = violation[np.isfinite(violation)]
        max_violation = (
            None
            if finite_violation.size == 0
            else float(np.max(finite_violation))
        )
        violation_count = int(np.count_nonzero(finite_violation > 0.0))
        worst_rows = _worst_constraint_rows(
            scaled_value=scaled_value,
            scaled_error=scaled_error,
            scaled_lower_bound=scaled_lower_bound,
            scaled_upper_bound=scaled_upper_bound,
        )
        return {
            "status": "passed",
            "reason": "constraint term evaluated",
            "name": type(constraint).__name__,
            "qualified_type": _qualified_type_name(constraint),
            "dim_f": int(scaled_value.size),
            "value_seconds": time.perf_counter() - start,
            "scaled_value_all_finite": value_all_finite,
            "scaled_error_all_finite": error_all_finite,
            "scaled_value_finite_count": _finite_count(scaled_value),
            "scaled_value_nonfinite_count": _nonfinite_count(scaled_value),
            "scaled_error_finite_count": _finite_count(scaled_error),
            "scaled_error_nonfinite_count": _nonfinite_count(scaled_error),
            "scaled_value_min": _finite_min(scaled_value),
            "scaled_value_max": _finite_max(scaled_value),
            "scaled_lower_bound_min": _finite_min(scaled_lower_bound),
            "scaled_upper_bound_max": _finite_max(scaled_upper_bound),
            "max_abs_scaled_error": max_violation,
            "violation_count": violation_count,
            "worst_rows": worst_rows,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "name": type(constraint).__name__,
            "qualified_type": _qualified_type_name(constraint),
            "dim_f": 0,
            "value_seconds": time.perf_counter() - start,
            "scaled_value_all_finite": False,
            "scaled_error_all_finite": False,
            "scaled_value_finite_count": 0,
            "scaled_value_nonfinite_count": 0,
            "scaled_error_finite_count": 0,
            "scaled_error_nonfinite_count": 0,
            "scaled_value_min": None,
            "scaled_value_max": None,
            "scaled_lower_bound_min": None,
            "scaled_upper_bound_max": None,
            "max_abs_scaled_error": None,
            "violation_count": 0,
            "worst_rows": [],
        }


def _build_constraint_term_if_needed(constraint: object) -> None:
    if bool(getattr(constraint, "built", getattr(constraint, "_built", False))):
        return
    build_method = getattr(constraint, "build")
    build_method(use_jit=False, verbose=0)


def _constraint_scaled_bounds(
    constraint: object,
    *,
    dim_f: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_bounds = getattr(constraint, "bounds", None)
    if raw_bounds is not None:
        lower_raw = np.ones(dim_f, dtype=float) * np.asarray(raw_bounds[0], dtype=float)
        upper_raw = np.ones(dim_f, dtype=float) * np.asarray(raw_bounds[1], dtype=float)
        return (
            _constraint_scale_bound(constraint, lower_raw),
            _constraint_scale_bound(constraint, upper_raw),
        )
    raw_target = getattr(constraint, "target")
    target_raw = np.ones(dim_f, dtype=float) * np.asarray(raw_target, dtype=float)
    target_scaled = _constraint_scale_bound(constraint, target_raw)
    return target_scaled, target_scaled


def _constraint_scale_bound(
    constraint: object,
    raw_value: np.ndarray,
) -> np.ndarray:
    scale_method = getattr(constraint, "_scale")
    scaled = np.asarray(scale_method(raw_value), dtype=float).reshape(-1)
    normalize_target = bool(getattr(constraint, "_normalize_target", True))
    if not normalize_target:
        normalization = np.asarray(getattr(constraint, "normalization"), dtype=float)
        scaled = scaled * normalization
    return scaled


def _worst_constraint_rows(
    *,
    scaled_value: np.ndarray,
    scaled_error: np.ndarray,
    scaled_lower_bound: np.ndarray,
    scaled_upper_bound: np.ndarray,
) -> list[dict[str, object]]:
    order_values = np.nan_to_num(
        np.abs(scaled_error),
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    if order_values.size == 0:
        return []
    ordered_indices = np.argsort(order_values)[::-1]
    rows: list[dict[str, object]] = []
    for index in ordered_indices[:_MAX_CONSTRAINT_FEASIBILITY_ROWS]:
        row_index = int(index)
        rows.append(
            {
                "index": row_index,
                "scaled_value": _json_float(scaled_value[row_index]),
                "scaled_error": _json_float(scaled_error[row_index]),
                "scaled_lower_bound": _json_float(scaled_lower_bound[row_index]),
                "scaled_upper_bound": _json_float(scaled_upper_bound[row_index]),
            }
        )
    return rows


def _finite_count(values: np.ndarray) -> int:
    return int(np.count_nonzero(np.isfinite(values)))


def _nonfinite_count(values: np.ndarray) -> int:
    return int(values.size - _finite_count(values))


def _finite_min(values: np.ndarray) -> float | None:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    return float(np.min(finite_values))


def _finite_max(values: np.ndarray) -> float | None:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    return float(np.max(finite_values))


def _json_float(value: object) -> float | None:
    normalized = float(value)
    if not math.isfinite(normalized):
        return None
    return normalized


def _max_optional_float(values: object) -> float | None:
    finite_values = [
        float(value)
        for value in values
        if isinstance(value, int | float) and math.isfinite(float(value))
    ]
    if not finite_values:
        return None
    return max(finite_values)


def _int_from_report(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _raise_if_optimizer_rejects_joint_constraints(
    *,
    optimizer_method: str,
    optimizer_registry: Mapping[str, Mapping[str, object]],
    optimizer_class: type[object],
    constraints: tuple[object, ...],
    desc_source_root: Path | None,
    desc_version: str | None,
    maxiter: int | None,
    verbose: int,
    optimizer_controls: DescOptimizerControls,
    allow_high_memory_optimizer: bool,
    objective_function: object,
    input_equilibrium: object,
    input_coilset: object,
) -> None:
    if len(constraints) == 0:
        return
    if _desc_optimizer_uses_proximal_constraint_wrapper(
        optimizer_method=optimizer_method,
        optimizer_wrappers=getattr(optimizer_class, "_wrappers", ()),
    ):
        if _constraints_are_proximal_equilibrium_constraints(constraints):
            return
        report = _joint_runtime_solve_report(
            status="failed",
            reason=(
                "DESC optimizer method "
                f"{optimizer_method!r} uses a proximal constraint wrapper, but "
                "joint DESC optimization assembled constraints that cannot be "
                "projected by DESC ProximalProjection. Use a DESC optimizer that "
                "handles equality constraints directly, or stage non-equilibrium "
                "constraints as objectives."
            ),
            desc_source_root=desc_source_root,
            desc_version=desc_version,
            optimizer_method=optimizer_method,
            maxiter=maxiter,
            verbose=verbose,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=allow_high_memory_optimizer,
            objective_function=objective_function,
            constraints=constraints,
            input_equilibrium=input_equilibrium,
            input_coilset=input_coilset,
            optimized_equilibrium=None,
            optimized_coilset=None,
            optimized_equilibrium_path=None,
            optimized_coilset_path=None,
            failed_optimizer_equilibrium_checkpoint_path=None,
            failed_optimizer_coilset_checkpoint_path=None,
            constraint_feasibility_report_path=None,
            optimizer_result=None,
        )
        raise DescJointRuntimeSolveError(report)
    if _desc_optimizer_supports_equality_constraints(
        optimizer_method=optimizer_method,
        optimizer_registry=optimizer_registry,
        optimizer_class=optimizer_class,
    ):
        return
    report = _joint_runtime_solve_report(
        status="failed",
        reason=(
            "DESC optimizer method "
            f"{optimizer_method!r} does not support equality constraints, but "
            "joint DESC optimization assembled hard constraints. Use a "
            "constraint-capable DESC optimizer method."
        ),
        desc_source_root=desc_source_root,
        desc_version=desc_version,
        optimizer_method=optimizer_method,
        maxiter=maxiter,
        verbose=verbose,
        optimizer_controls=optimizer_controls,
        allow_high_memory_optimizer=allow_high_memory_optimizer,
        objective_function=objective_function,
        constraints=constraints,
        input_equilibrium=input_equilibrium,
        input_coilset=input_coilset,
        optimized_equilibrium=None,
        optimized_coilset=None,
        optimized_equilibrium_path=None,
        optimized_coilset_path=None,
        failed_optimizer_equilibrium_checkpoint_path=None,
        failed_optimizer_coilset_checkpoint_path=None,
        constraint_feasibility_report_path=None,
        optimizer_result=None,
    )
    raise DescJointRuntimeSolveError(report)


def _raise_if_optimizer_rejects_proximal_options(
    *,
    optimizer_method: str,
    optimizer_class: type[object],
    optimizer_controls: DescOptimizerControls,
) -> None:
    if not optimizer_controls.has_proximal_options():
        return
    if _desc_optimizer_uses_proximal_constraint_wrapper(
        optimizer_method=optimizer_method,
        optimizer_wrappers=getattr(optimizer_class, "_wrappers", ()),
    ):
        return
    raise ValueError(
        "DESC proximal optimizer controls require a prox-/proximal- optimizer "
        f"method; got {optimizer_method!r}."
    )


def _desc_optimizer_supports_equality_constraints(
    *,
    optimizer_method: str,
    optimizer_registry: Mapping[str, Mapping[str, object]],
    optimizer_class: type[object],
) -> bool:
    optimizer_wrappers = getattr(optimizer_class, "_wrappers", ())
    registry_key = _desc_optimizer_registry_key(
        optimizer_method=optimizer_method,
        optimizer_wrappers=optimizer_wrappers,
    )
    optimizer_spec = optimizer_registry.get(registry_key)
    if optimizer_spec is None:
        raise ValueError(
            f"DESC optimizer registry does not contain method {registry_key!r}."
        )
    if bool(optimizer_spec.get("equality_constraints", False)):
        return True
    return False


def _desc_optimizer_uses_proximal_constraint_wrapper(
    *,
    optimizer_method: str,
    optimizer_wrappers: object,
) -> bool:
    if not isinstance(optimizer_wrappers, Sequence) or isinstance(
        optimizer_wrappers,
        (str, bytes),
    ):
        return False
    for wrapper in optimizer_wrappers:
        if not isinstance(wrapper, str) or wrapper.lower() not in {"prox", "proximal"}:
            continue
        if optimizer_method.lower().startswith(f"{wrapper.lower()}-"):
            return True
    return False


def _constraints_are_proximal_equilibrium_constraints(
    constraints: tuple[object, ...],
) -> bool:
    return all(
        bool(getattr(constraint, "_equilibrium", False))
        or bool(getattr(constraint, "linear", getattr(constraint, "_linear", False)))
        for constraint in constraints
    )


def _desc_optimizer_registry_key(
    *,
    optimizer_method: str,
    optimizer_wrappers: object,
) -> str:
    if isinstance(optimizer_wrappers, Sequence) and not isinstance(
        optimizer_wrappers,
        (str, bytes),
    ):
        for wrapper in optimizer_wrappers:
            if not isinstance(wrapper, str) or wrapper == "":
                continue
            wrapper_prefix = f"{wrapper}-"
            if optimizer_method.lower().startswith(wrapper_prefix):
                return optimizer_method[len(wrapper_prefix) :]
    return optimizer_method


def _single_optimized_thing(optimized_things: object) -> object:
    if isinstance(optimized_things, Sequence) and not isinstance(
        optimized_things,
        (str, bytes),
    ):
        if len(optimized_things) != 1:
            raise ValueError(
                "DESC fixed-equilibrium polish expected one optimized CoilSet; "
                f"got {len(optimized_things)} objects."
            )
        return optimized_things[0]
    raise TypeError("DESC optimizer must return a one-element optimized-things list.")


def _joint_optimizer_things(
    *,
    objective_function: object,
    constraints: tuple[object, ...],
    equilibrium: object,
    coilset: object,
) -> tuple[object, object]:
    ordered_things: list[object] = []
    for term in (*constraints, objective_function):
        _append_joint_term_things(
            ordered_things,
            term,
            equilibrium=equilibrium,
            coilset=coilset,
        )
    for thing in (equilibrium, coilset):
        _append_unique_identity(ordered_things, thing)
    if len(ordered_things) != 2:
        raise ValueError("DESC joint optimization requires equilibrium and coilset.")
    return ordered_things[0], ordered_things[1]


def _append_joint_term_things(
    ordered_things: list[object],
    term: object,
    *,
    equilibrium: object,
    coilset: object,
) -> None:
    term_things = getattr(term, "things", ())
    if not isinstance(term_things, Sequence) or isinstance(
        term_things,
        (str, bytes),
    ):
        return
    for thing in term_things:
        if thing is equilibrium or thing is coilset:
            _append_unique_identity(ordered_things, thing)


def _append_unique_identity(ordered_things: list[object], thing: object) -> None:
    if not any(existing is thing for existing in ordered_things):
        ordered_things.append(thing)


def _joint_optimized_things(
    optimized_things: object,
    *,
    input_equilibrium: object,
    input_coilset: object,
) -> tuple[object, object]:
    if isinstance(optimized_things, Sequence) and not isinstance(
        optimized_things,
        (str, bytes),
    ):
        if len(optimized_things) != 2:
            raise ValueError(
                "DESC joint optimization expected optimized Equilibrium and "
                f"CoilSet; got {len(optimized_things)} objects."
            )
        return (
            _select_optimized_thing(
                optimized_things,
                input_thing=input_equilibrium,
                thing_label="Equilibrium",
            ),
            _select_optimized_thing(
                optimized_things,
                input_thing=input_coilset,
                thing_label="CoilSet",
            ),
        )
    raise TypeError(
        "DESC optimizer must return a two-object optimized-things sequence."
    )


def _select_optimized_thing(
    optimized_things: Sequence[object],
    *,
    input_thing: object,
    thing_label: str,
) -> object:
    matches = tuple(
        thing for thing in optimized_things if isinstance(thing, type(input_thing))
    )
    if len(matches) != 1:
        raise TypeError(
            f"DESC joint optimization expected one optimized {thing_label} "
            f"matching {type(input_thing).__name__}; got {len(matches)}."
        )
    return matches[0]


def _save_desc_equilibrium(equilibrium: object, path: Path) -> None:
    _save_desc_saveables_atomically(
        (
            _DescSaveRequest(
                saveable=equilibrium,
                final_path=path,
                object_label="Optimized DESC Equilibrium",
            ),
        ),
    )


def _save_desc_coilset(coilset: object, path: Path) -> None:
    _save_desc_saveables_atomically(
        (
            _DescSaveRequest(
                saveable=coilset,
                final_path=path,
                object_label="Optimized DESC CoilSet",
            ),
        ),
    )


def _save_desc_equilibrium_and_coilset(
    *,
    optimized_equilibrium: object,
    equilibrium_path: Path,
    optimized_coilset: object,
    coilset_path: Path,
) -> None:
    _save_desc_saveables_atomically(
        (
            _DescSaveRequest(
                saveable=optimized_equilibrium,
                final_path=equilibrium_path,
                object_label="Optimized DESC Equilibrium",
            ),
            _DescSaveRequest(
                saveable=optimized_coilset,
                final_path=coilset_path,
                object_label="Optimized DESC CoilSet",
            ),
        ),
    )


def _save_desc_saveables_atomically(save_requests: Sequence[_DescSaveRequest]) -> None:
    staged_paths: list[tuple[Path, Path]] = []
    try:
        for save_request in save_requests:
            temporary_path = _desc_save_temporary_path(save_request.final_path)
            temporary_path.unlink(missing_ok=True)
            staged_paths.append((temporary_path, save_request.final_path))
            _save_desc_saveable(
                save_request.saveable,
                temporary_path,
                object_label=save_request.object_label,
            )
        for temporary_path, final_path in staged_paths:
            os.replace(temporary_path, final_path)
    except Exception:
        for temporary_path, _ in staged_paths:
            temporary_path.unlink(missing_ok=True)
        raise


def _save_desc_saveable(saveable: object, path: Path, *, object_label: str) -> None:
    save_method = getattr(saveable, "save", None)
    if not callable(save_method):
        raise TypeError(f"{object_label} does not provide a save method.")
    save_method(os.fspath(path))
    if not path.is_file():
        raise FileNotFoundError(f"{object_label} save did not write {path}.")


def _desc_save_temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def desc_high_memory_optimizer_blocked_reason(lane_name: str) -> str:
    return (
        f"DESC {lane_name} optimizer execution is blocked by default because "
        "DESC's combined ObjectiveFunction optimizer path materializes "
        "high-memory derivative state for real banana seeds. Pass "
        "--allow-high-memory-desc-optimizer only in a resource-managed "
        "environment after acknowledging that risk."
    )


def build_desc_optimizer_controls(
    *,
    ftol: float | None = None,
    xtol: float | None = None,
    gtol: float | None = None,
    ctol: float | None = None,
    max_nfev: int | None = None,
    max_dx: float | None = None,
    initial_trust_radius: float | None = None,
    max_trust_radius: float | None = None,
    min_trust_radius: float | None = None,
    proximal_perturb_order: int | None = None,
    proximal_solve_maxiter: int | None = None,
    proximal_solve_during_build: bool | None = None,
) -> DescOptimizerControls:
    """Build validated controls for DESC Optimizer.optimize."""

    return DescOptimizerControls(
        ftol=_validate_optional_positive_float(ftol, field_name="ftol"),
        xtol=_validate_optional_positive_float(xtol, field_name="xtol"),
        gtol=_validate_optional_positive_float(gtol, field_name="gtol"),
        ctol=_validate_optional_positive_float(ctol, field_name="ctol"),
        max_nfev=_normalize_optional_positive_int(
            max_nfev,
            field_name="max_nfev",
        ),
        max_dx=_validate_optional_positive_float(max_dx, field_name="max_dx"),
        initial_trust_radius=_validate_optional_positive_float(
            initial_trust_radius,
            field_name="initial_trust_radius",
        ),
        max_trust_radius=_validate_optional_positive_float(
            max_trust_radius,
            field_name="max_trust_radius",
        ),
        min_trust_radius=_validate_optional_nonnegative_float(
            min_trust_radius,
            field_name="min_trust_radius",
        ),
        proximal_perturb_order=_normalize_optional_positive_int(
            proximal_perturb_order,
            field_name="proximal_perturb_order",
        ),
        proximal_solve_maxiter=_normalize_optional_positive_int(
            proximal_solve_maxiter,
            field_name="proximal_solve_maxiter",
        ),
        proximal_solve_during_build=_validate_optional_bool(
            proximal_solve_during_build,
            field_name="proximal_solve_during_build",
        ),
    )


def _coerce_optimizer_controls(
    optimizer_controls: DescOptimizerControls | None,
) -> DescOptimizerControls:
    if optimizer_controls is None:
        return build_desc_optimizer_controls()
    if not isinstance(optimizer_controls, DescOptimizerControls):
        raise ValueError("optimizer_controls must be a DescOptimizerControls instance.")
    return optimizer_controls


def _validate_optional_positive_float(
    value: float | None,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a positive finite float or None.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite float or None.")
    return normalized


def _validate_optional_nonnegative_float(
    value: float | None,
    *,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a nonnegative finite float or None.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be a nonnegative finite float or None.")
    return normalized


def _normalize_optional_positive_int(
    value: int | None,
    *,
    field_name: str,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer or None.")
    return value


def _validate_optional_bool(
    value: bool | None,
    *,
    field_name: str,
) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean or None.")
    return value


def _optimizer_success(optimizer_result: object | None) -> bool | None:
    value = _optimizer_value(optimizer_result, "success")
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def _optimizer_int(optimizer_result: object | None, key: str) -> int | None:
    value = _optimizer_value(optimizer_result, key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optimizer_message(optimizer_result: object | None) -> str | None:
    value = _optimizer_value(optimizer_result, "message")
    if value is None:
        return None
    return str(value)


def _optimizer_value(optimizer_result: object | None, key: str) -> object | None:
    if optimizer_result is None:
        return None
    if isinstance(optimizer_result, Mapping):
        return optimizer_result.get(key)
    return getattr(optimizer_result, key, None)


def _validate_optimizer_method(value: str) -> None:
    if not isinstance(value, str) or value == "":
        raise ValueError("DESC optimizer method must be a nonempty string.")


def _validate_optional_positive_int(value: int | None, *, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer or None.")


def _validate_nonnegative_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer.")


def _validate_bool(value: bool, *, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool.")


def _qualified_type_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _desc_version(desc_module: object) -> str | None:
    version = getattr(desc_module, "__version__", None)
    if isinstance(version, str) and version != "":
        return version
    return None


__all__ = [
    "DescFixedPolishRuntimeSolveError",
    "DescFixedPolishRuntimeSolveReport",
    "DescFixedPolishRuntimeSolveResult",
    "DescJointRuntimeSolveError",
    "DescJointRuntimeSolveReport",
    "DescJointRuntimeSolveResult",
    "DescOptimizerControls",
    "build_desc_optimizer_controls",
    "desc_fixed_equilibrium_polish_setup_failure_report",
    "desc_high_memory_optimizer_blocked_reason",
    "desc_joint_optimization_setup_failure_report",
    "run_desc_fixed_equilibrium_polish_runtime",
    "run_desc_joint_optimization_runtime",
]
