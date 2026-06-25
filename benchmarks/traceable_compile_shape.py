"""StableHLO shape summaries for traceable JAX callables."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import jax


@dataclass(frozen=True)
class LoweringMeasurement:
    """Textual lowering payload plus elapsed lowering time."""

    lowered_text: str
    lower_s: float


def lower_to_text(fn: Callable[..., Any], *args: Any) -> LoweringMeasurement:
    """Lower a JAX callable and return its textual compiler IR."""
    lowerable_fn = fn if hasattr(fn, "lower") else jax.jit(fn)
    start = time.perf_counter()
    lowered_text = lowerable_fn.lower(*args).as_text()
    return LoweringMeasurement(
        lowered_text=lowered_text,
        lower_s=time.perf_counter() - start,
    )


def summarize_lowered_text(
    label: str,
    lowered_text: str,
    *,
    lower_s: float | None = None,
) -> dict[str, int | float | str | None]:
    """Count staged control-flow markers in textual StableHLO/MHLO IR."""
    return {
        "label": label,
        "lower_s": lower_s,
        "text_bytes": len(lowered_text.encode("utf-8")),
        "text_lines": len(lowered_text.splitlines()),
        "stablehlo_while_count": lowered_text.count("stablehlo.while"),
        "stablehlo_case_count": lowered_text.count("stablehlo.case"),
        "stablehlo_if_count": lowered_text.count("stablehlo.if"),
        "mhlo_while_count": lowered_text.count("mhlo.while"),
        "mhlo_case_count": lowered_text.count("mhlo.case"),
        "mhlo_if_count": lowered_text.count("mhlo.if"),
        "jax_cond_source_count": lowered_text.count("lax.cond"),
        "jax_while_source_count": lowered_text.count("lax.while_loop"),
    }


def summarize_lowered_callable(
    label: str,
    fn: Callable[..., Any],
    *args: Any,
) -> dict[str, int | float | str | None]:
    """Lower a callable and summarize its staged compiler shape."""
    measurement = lower_to_text(fn, *args)
    return summarize_lowered_text(
        label,
        measurement.lowered_text,
        lower_s=measurement.lower_s,
    )
