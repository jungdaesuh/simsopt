"""
solver_log.py
─────────────
Fold simsopt's Boozer-solver stdout into this driver's log stream instead of
letting BFGS/Newton progress lines interleave raw with everything else, and
scrape the inner iteration counts out of those lines so each outer optimizer
evaluation can report how many inner solver iterations it took.

Ported from banana_drivers' ``utils/io.py`` (``ErrStream``, ``LogStream``,
``stdout_to_log``) and ``drivers/singlestage.py`` (``BFGS_PAT``,
``NEWTON_PAT``, ``make_solver_iter_tap``). The upstream ``stdout_to_log``
took a ``DriverLog`` object and called ``log(line)`` / ``log.write_raw(text)``
on it; this repo has no ``DriverLog`` — drivers log through a module-level
``proc0_print`` function instead — so ``stdout_to_log`` here takes a plain
line-sink callable, ``sink: Callable[[str], None]``, and every former
``write_raw(text)`` call becomes ``sink(text)``: ``LogStream`` still calls
``sink`` once per complete buffered line (unchanged from the source), while
``ErrStream`` and the traceback handler call ``sink`` with the raw
(possibly multi-line, possibly partial) chunk exactly as the source wrote it
to file, unbuffered.

Used by:
- 03_singlestage_driver.py  (fold BoozerLS solver stdout into the driver log)
"""
import contextlib
import io
import re
import sys
import threading
import traceback


BFGS_PAT = re.compile(r"\b(?:L-BFGS-B|BFGS|LBFGS) solve - .*\biter=(\d+)")
NEWTON_PAT = re.compile(r"\bNEWTON solve - .*\biter=(\d+)")
_STDOUT_CAPTURE_LOCK = threading.RLock()


class ErrStream:
    """Tee stderr writes to the real stderr and to ``sink``."""

    def __init__(self, sink, stderr):
        self._sink = sink
        self._stderr = stderr

    def write(self, s):
        self._stderr.write(s)
        self._sink(s)
        return len(s)

    def flush(self):
        self._stderr.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self._stderr.fileno()


class LogStream:
    """Redirect target for stdout that line-buffers writes into ``sink``."""

    def __init__(self, sink, *, tap=None):
        self._sink = sink
        self._buffer = ""
        self._tap = tap

    def write(self, s):
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._sink(line)
            if self._tap is not None:
                self._tap(line)
        return len(s)

    def flush(self):
        if self._buffer:
            self._sink(self._buffer)
            if self._tap is not None:
                self._tap(self._buffer)
            self._buffer = ""

    def isatty(self):
        return False

    def fileno(self):
        raise io.UnsupportedOperation("fileno")


@contextlib.contextmanager
def stdout_to_log(sink, *, tap=None, capture_stderr=True):
    """Redirect stdout (and, by default, stderr) into ``sink`` for the block.

    ``sink`` is a plain line-sink callable, ``Callable[[str], None]``, in
    place of the upstream ``DriverLog``. Pass a solver-iteration ``tap`` (see
    ``make_solver_iter_tap``) to also scrape inner iteration counts out of
    the captured stdout lines.

    Upstream, ``DriverLog`` prints to the ``sys.__stdout__``/``sys.stdout``
    it captured at construction time, so its calls never route back through
    the redirected stream it is logging on behalf of. A plain callable has
    no such built-in protection: this repo's real sink, ``proc0_print``, is
    just ``print(*args, flush=True)``, which writes to whatever
    ``sys.stdout`` currently is — and while this context manager is active,
    that *is* the ``LogStream`` calling ``sink`` in the first place. Calling
    an unshielded ``sink`` from inside ``LogStream``/``ErrStream`` would
    therefore recurse without bound. To restore that lost protection without
    pushing it onto every caller, the real ``sys.stdout``/``sys.stderr`` are
    snapshotted here (the *current* streams at entry, not
    ``sys.__stdout__`` — this must also work nested, and under pytest's own
    capture) and ``sink`` is wrapped so each call temporarily reinstates
    them, then restores the redirect for the rest of the block once the
    call returns.
    """
    # ``sys.stdout`` and ``sys.stderr`` are process-global. Serializing the
    # capture contexts prevents two helper users from routing each other's
    # solver output to the wrong sink; nested use in one thread remains valid.
    with _STDOUT_CAPTURE_LOCK:
        real_stdout = sys.stdout
        real_stderr = sys.stderr

        def shielded_sink(text):
            redirected_stdout, redirected_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = real_stdout, real_stderr
            try:
                sink(text)
            finally:
                sys.stdout, sys.stderr = redirected_stdout, redirected_stderr

        out_stream = LogStream(shielded_sink, tap=tap)
        err_redirect = (
            contextlib.redirect_stderr(ErrStream(shielded_sink, real_stderr))
            if capture_stderr
            else contextlib.nullcontext()
        )
        try:
            with contextlib.redirect_stdout(out_stream), err_redirect:
                try:
                    yield
                finally:
                    out_stream.flush()
        except BaseException:
            shielded_sink(traceback.format_exc())
            raise


def make_solver_iter_tap(solver_iters):
    """Return a ``LogStream`` tap that scrapes inner BFGS/Newton iteration counts.

    Only updates ``solver_iters[key]`` when ``key`` is already present in the
    dict, so e.g. a BoozerExact run with no ``bfgs_nit`` key stays untouched.
    """

    def tap(line):
        m = BFGS_PAT.search(line)
        if m and "bfgs_nit" in solver_iters:
            solver_iters["bfgs_nit"] = int(m.group(1))
        m = NEWTON_PAT.search(line)
        if m and "newton_nit" in solver_iters:
            solver_iters["newton_nit"] = int(m.group(1))
    return tap
