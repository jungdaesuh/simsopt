"""Locate shared unittest helpers for descendant pytest modules."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.register_assert_rewrite("jax_test_support")
