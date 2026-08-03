import io
import sys
import threading
from pathlib import Path

import pytest


HBT_DIR = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "HBT_BANANA"
)
sys.path.insert(0, str(HBT_DIR / "utils"))

from solver_log import (  # noqa: E402
    ErrStream,
    LogStream,
    make_solver_iter_tap,
    stdout_to_log,
)


def test_logstream_captures_multi_line_write_as_separate_sink_calls():
    lines = []
    stream = LogStream(lines.append)

    stream.write("first\nsecond\n")

    assert lines == ["first", "second"], (
        "LogStream must split a multi-line write into one sink call per line"
    )


def test_logstream_buffers_write_split_mid_line_across_two_calls():
    lines = []
    stream = LogStream(lines.append)

    stream.write("hel")
    stream.write("lo world\n")

    assert lines == ["hello world"], (
        "LogStream must buffer a write split mid-line until the newline arrives"
    )


def test_logstream_flush_emits_trailing_partial_line():
    lines = []
    stream = LogStream(lines.append)

    stream.write("no newline yet")
    assert lines == [], "a partial line must not be logged before flush() is called"

    stream.flush()

    assert lines == ["no newline yet"], (
        "flush() must emit the buffered trailing partial line"
    )


def test_logstream_flush_feeds_trailing_partial_line_to_tap():
    lines = []
    solver_iters = {"bfgs_nit": None}
    stream = LogStream(
        lines.append,
        tap=make_solver_iter_tap(solver_iters),
    )

    stream.write("L-BFGS-B solve - it=1 iter=23 residual=1e-3")
    stream.flush()

    assert lines == ["L-BFGS-B solve - it=1 iter=23 residual=1e-3"]
    assert solver_iters["bfgs_nit"] == 23, (
        "flush() must pass a trailing solver line through the iteration tap"
    )


def test_logstream_isatty_and_fileno_match_source_contract():
    stream = LogStream(lambda line: None)

    assert stream.isatty() is False, "LogStream.isatty() must report False"
    with pytest.raises(io.UnsupportedOperation):
        stream.fileno()


def test_make_solver_iter_tap_extracts_bfgs_and_newton_iteration_counts():
    solver_iters = {"bfgs_nit": None, "newton_nit": None}
    tap = make_solver_iter_tap(solver_iters)

    tap("L-BFGS-B solve - it=3 iter=17 residual=1e-9")
    tap("NEWTON solve - iter=4 residual=2e-12")

    assert solver_iters == {"bfgs_nit": 17, "newton_nit": 4}, (
        "tap must scrape the trailing iter=<n> count out of realistic "
        "BFGS/Newton solver lines"
    )


def test_make_solver_iter_tap_leaves_absent_keys_untouched():
    solver_iters = {"newton_nit": None}
    tap = make_solver_iter_tap(solver_iters)

    tap("L-BFGS-B solve - it=3 iter=17 residual=1e-9")

    assert "bfgs_nit" not in solver_iters, (
        "tap must not add bfgs_nit when it was not already a key "
        "(e.g. a BoozerExact run that never reports BFGS iterations)"
    )


def test_errstream_write_reaches_real_stderr_and_sink():
    fake_stderr = io.StringIO()
    sunk = []
    err = ErrStream(sunk.append, fake_stderr)

    err.write("warning: something\n")

    assert fake_stderr.getvalue() == "warning: something\n", (
        "ErrStream must still forward writes to the real stderr"
    )
    assert sunk == ["warning: something\n"], (
        "ErrStream must also forward the raw chunk to sink"
    )


def test_stdout_to_log_logs_traceback_before_reraising():
    sunk = []

    with pytest.raises(ValueError):
        with stdout_to_log(sunk.append):
            raise ValueError("boom")

    assert any("Traceback" in chunk and "boom" in chunk for chunk in sunk), (
        "stdout_to_log must write the formatted traceback to sink before re-raising"
    )


def test_stdout_to_log_folds_print_output_and_feeds_tap():
    lines = []
    solver_iters = {"bfgs_nit": None}
    tap = make_solver_iter_tap(solver_iters)

    with stdout_to_log(lines.append, tap=tap):
        print("L-BFGS-B solve - it=1 iter=5 residual=1e-3")

    assert lines == ["L-BFGS-B solve - it=1 iter=5 residual=1e-3"], (
        "stdout_to_log must fold print() output into sink line by line"
    )
    assert solver_iters["bfgs_nit"] == 5, (
        "stdout_to_log must wire the tap into the redirected stdout so "
        "LogStream feeds it captured lines"
    )


def test_stdout_to_log_shields_print_based_sink_from_infinite_recursion():
    """Regression test: a faithful stand-in for the real sink (this repo's
    ``proc0_print``, i.e. a function that itself calls ``print()``) must not
    recurse through the very stdout redirect it is being invoked under.

    A list-appending sink cannot reproduce this bug, because it never writes
    back to ``sys.stdout`` — this is exactly why the original 9-test suite
    missed it.
    """

    def proc0_print_like_sink(text):
        print(text)

    real_stdout = sys.stdout
    fake_real_stdout = io.StringIO()
    sys.stdout = fake_real_stdout
    try:
        with stdout_to_log(proc0_print_like_sink):
            print("L-BFGS-B solve - it=1 iter=5 residual=1e-3")
    finally:
        sys.stdout = real_stdout

    assert fake_real_stdout.getvalue() == (
        "L-BFGS-B solve - it=1 iter=5 residual=1e-3\n"
    ), (
        "a print()-based sink must write to the real stdout snapshotted at "
        "entry instead of recursing back through the redirected LogStream"
    )


def test_stdout_to_log_reinstates_capture_after_each_shielded_sink_call():
    """Capture must resume for the rest of the block after a sink call, not
    be permanently dropped once the first shielded call returns."""
    lines = []

    def echoing_sink(text):
        lines.append(text)
        print(f"echoed: {text}")

    real_stdout = sys.stdout
    fake_real_stdout = io.StringIO()
    sys.stdout = fake_real_stdout
    try:
        with stdout_to_log(echoing_sink):
            print("first")
            print("second")
    finally:
        sys.stdout = real_stdout

    assert lines == ["first", "second"], (
        "LogStream capture must still be active for the second print() "
        "after the first shielded sink call returns"
    )
    expected_echo = "\n".join(["echoed: first", "echoed: second"]) + "\n"
    assert fake_real_stdout.getvalue() == expected_echo, (
        "each shielded sink call must reach the real stdout without "
        "recursing, for every call in the block, not just the first"
    )


def test_stdout_to_log_serializes_concurrent_capture_contexts():
    first_lines = []
    second_lines = []
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()

    def first_worker():
        with stdout_to_log(first_lines.append):
            print("first")
            first_entered.set()
            release_first.wait(timeout=2.0)

    def second_worker():
        with stdout_to_log(second_lines.append):
            second_entered.set()
            print("second")

    first = threading.Thread(target=first_worker)
    second = threading.Thread(target=second_worker)
    first.start()
    assert first_entered.wait(timeout=2.0)
    second.start()
    try:
        assert not second_entered.wait(timeout=0.05), (
            "a second capture context must wait while the process-global "
            "stdout redirect is owned by the first context"
        )
    finally:
        release_first.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)

    assert not first.is_alive() and not second.is_alive()
    assert first_lines == ["first"]
    assert second_lines == ["second"]
