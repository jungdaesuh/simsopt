"""Materialize a rerunnable VMEC input file from a saved ``wout`` file.

Plan section "SIMSOPT/VMEC API Constraints":

    "Vmec(wout_filename) is valid for read-only inspection of a saved wout,
    but a Vmec object initialized from wout is not rerunnable because the
    original VMEC input/radial multigrid data is not available."

    "Optimization seeds must therefore materialize a rerunnable VMEC input
    file, not only a copied wout."

This module is the single seed-materialization path. The loader:

1. Locates the ``input.<ext>`` paired with the saved ``wout_<ext>.nc``.
2. Rejects the seed when ``lfreeb=True`` or ``mgrid_file`` is populated
   (plan section "SIMSOPT/VMEC API Constraints" H9: "reject any seed input
   with ``lfreeb = .true.`` or a populated ``mgrid_file``").
3. Pins ``lfreeb=False``, pressure/current profiles to zero (Stage A:
   fixed-boundary vacuum only).
4. Calls ``vmec.write_input(out_path)`` to produce a rerunnable namelist.

Hand-editing ``indata.rbc/rbs/zbc/zbs`` is forbidden per plan: VMEC boundary
coefficients are owned by the SIMSOPT surface object and silently clobbered
by ``set_indata()`` on the next ``run()``.
"""

from __future__ import annotations

from pathlib import Path

from simsopt.mhd import Vmec


def materialize_rerunnable_input(
    wout_path: Path,
    *,
    out_path: Path,
    lfreeb: bool = False,
    pres_scale: float = 0.0,
) -> Path:
    """Materialize a rerunnable ``input.<ext>`` from a saved ``wout``.

    Args:
        wout_path: Path to a ``wout_*.nc`` VMEC output file.
        out_path: Destination path for the materialized VMEC input.
        lfreeb: Free-boundary flag. Must be ``False`` for Stage A. The
            parameter is exposed only to make the explicit pin visible at
            call sites; passing ``True`` raises ``ValueError``.
        pres_scale: Pressure scale. Must be ``0.0`` for Stage A (vacuum).

    Returns:
        ``out_path`` after the input file has been written.

    Raises:
        ValueError: If ``wout_path`` is not a ``wout_*.nc`` file, if the
            paired input reports ``lfreeb=True``, if it carries a populated
            ``mgrid_file``, or if the caller requests ``lfreeb=True`` or
            ``pres_scale != 0.0``.
    """

    wout_path = Path(wout_path)
    out_path = Path(out_path)
    if not wout_path.exists():
        raise FileNotFoundError(f"wout file not found: {wout_path}")
    if not wout_path.name.startswith("wout"):
        raise ValueError(
            f"Expected a wout_*.nc seed; got {wout_path.name!r}"
        )

    if lfreeb:
        raise ValueError(
            "Stage A is fixed-boundary; lfreeb=True is forbidden."
        )
    if pres_scale != 0.0:
        raise ValueError(
            "Stage A is vacuum-only; pres_scale must equal 0.0."
        )

    # SIMSOPT Vmec(wout) yields a non-runnable instance whose indata block
    # is *not* populated from the wout. To materialize a rerunnable input,
    # we need the original input data. Per plan: "Optimization seeds must
    # therefore materialize a rerunnable VMEC input file."
    #
    # If the wout was produced by an earlier SIMSOPT run via ``write_input``,
    # the radial multigrid block is implicit and we reconstruct it by
    # loading the wout's *paired* input file when present, then forcing the
    # fixed-boundary + vacuum pins. If no paired input exists, the caller
    # must supply one separately; this loader does not invent radial
    # schedules.
    paired_input = _find_paired_input(wout_path)
    if paired_input is None:
        raise ValueError(
            f"No paired input file alongside {wout_path}; cannot "
            "materialize a rerunnable seed from a wout alone. The plan "
            'requires "the original VMEC input/radial multigrid data" '
            "to be present."
        )

    vmec = Vmec(str(paired_input), verbose=False)
    vi = vmec.indata
    if bool(vi.lfreeb):
        raise ValueError(
            f"Seed input {paired_input} enables free boundary "
            "(lfreeb=.true.); reject."
        )
    mgrid = vi.mgrid_file.decode("utf-8").strip()
    has_mgrid = bool(mgrid) and mgrid.upper() != "NONE"
    if has_mgrid:
        raise ValueError(
            f"Seed input {paired_input} carries a populated mgrid_file "
            f"({mgrid!r}); reject."
        )

    # Pin fixed-boundary + vacuum even if the source already does.
    vi.lfreeb = False
    vi.pres_scale = 0.0
    vi.curtor = 0.0
    vi.ac[:] = 0.0
    vi.am[:] = 0.0

    vmec.write_input(str(out_path))
    return out_path


def _find_paired_input(wout_path: Path) -> Path | None:
    """Locate the ``input.<ext>`` that produced ``wout_<ext>.nc``.

    Convention: a wout named ``wout_<extension>.nc`` is produced from
    ``input.<extension>`` in the same directory. Returns ``None`` when no
    paired input is found.
    """

    # ``wout_<ext>.nc`` -> ``input.<ext>``.
    name = wout_path.name
    if not name.startswith("wout_") or not name.endswith(".nc"):
        return None
    extension = name[len("wout_") : -len(".nc")]
    candidate = wout_path.parent / f"input.{extension}"
    return candidate if candidate.exists() else None
