"""Lightweight execution-scale type shared across runtime boundaries."""

from __future__ import annotations

from typing import Literal

ExecutionScale = Literal["bounded", "native_default"]

__all__ = ("ExecutionScale",)
