"""Nested Boozer-surface family solver with optional fail-closed truncation.

Factors the outer-to-inner continuation march that
``initialize_published_surface_data_from_stage2_seed`` performs in the driver so
the same nested-family solve can be reused with two acceptance policies:

* ``allow_truncation=False`` (strict, the default): solve every requested shell
  and re-raise on the first solver failure, exactly as the driver's verbatim
  ``initialize_boozer_surface`` calls do today. This branch is a line-for-line
  mirror of ``initialize_boozer_surface`` (handoff ``:1523``) -- it calls the
  non-raising ``attempt_initialize_boozer_surface`` and reproduces its
  ``if not result.success: print(...); raise`` block -- so the published
  multisurface continuation stays byte-identical. The caller still runs the
  strict ``_require_published_surface_data_postconditions`` afterward.
* ``allow_truncation=True``: accept a shell only when it solved AND nests inside
  the previous accepted shell AND keeps volumes strictly ordered; stop at the
  first shell that fails any of those. The kept band is the OUTER suffix of the
  requested stack (e.g. ``{inner2, inner3, outer}``), so the inner0-based name
  postcondition cannot apply -- a relaxed nesting + volume-order check runs
  instead. Raise only when fewer than ``min_surfaces`` shells are accepted.

The march warm-starts each inner shell from the previous solved shell's iota/G
(read inline from ``prev.res``) and contracts its surface toward the next target
volume via the injected ``contract`` callable. Two driver-local helpers are
injected rather than imported because the driver runs as ``__main__``:
``contract`` (``contract_surface_to_target_volume``) and ``make_entry``
(``_surface_data_entry``).

``ordered_configs`` is the inner-to-outer config list (``inner0 .. innerN-2,
outer``); the outer shell is ``ordered_configs[-1]`` and is solved first from
``outer_seed``, then the inner shells are marched ``reversed(ordered_configs[:-1])``.
Returns ``(ordered_surface_data, provenance)`` where ``ordered_surface_data`` is
the accepted band re-ordered inner-to-outer and ``provenance`` records the
truncation outcome.
"""

import numpy as np

from banana_opt.single_stage_geometry import cross_sections_are_nested
from banana_opt.stage2_single_stage_handoff import attempt_initialize_boozer_surface


class BoozerFamilyOuterSeed:
    """Pre-validated warm start for the outermost shell of the family.

    ``surface`` is the seed surface used both as the solve geometry and the
    initial guess; ``iota`` and ``G`` are the explicit scalars the caller has
    already resolved (the driver validates ``G`` via ``_require_finite_explicit_G``
    before constructing this seed, preserving the outer-G finiteness raise).
    """

    def __init__(self, surface, iota, G):
        self.surface = surface
        self.iota = iota
        self.G = G


class BoozerFamilyProvenance:
    """Outcome record for a family solve.

    ``accepted`` is the number of shells kept; ``requested`` the number asked
    for; ``last_good_name`` the name of the innermost accepted shell (the march
    accepts outer-to-inner, so this is the last shell added before any stop);
    ``truncated`` is True iff fewer shells were accepted than requested.
    """

    def __init__(self, *, accepted, requested, last_good_name, truncated):
        self.accepted = accepted
        self.requested = requested
        self.last_good_name = last_good_name
        self.truncated = truncated


def _solve_or_raise(
    surf_prev,
    *,
    mpol,
    ntor,
    bs,
    vol_target,
    constraint_weight,
    iota,
    G,
    boozer_I,
    initial_surface_guess,
    nfp,
):
    """Solve one shell and re-raise on failure, mirroring ``initialize_boozer_surface``.

    Verbatim copy of ``initialize_boozer_surface``'s body (handoff ``:1551-1583``):
    call the non-raising attempt, return the surface on success, else print the
    same diagnostics and raise the same ``RuntimeError``. Keeps the strict family
    path byte-identical to the driver's existing continuation calls.
    """
    result = attempt_initialize_boozer_surface(
        surf_prev,
        mpol,
        ntor,
        bs,
        vol_target,
        constraint_weight,
        iota,
        G,
        boozer_I,
        initial_surface_guess=initial_surface_guess,
        nfp=nfp,
    )
    if result.success:
        return result.boozer_surface

    print(
        "Boozer initialization failed: "
        f"solve_success={result.solve_success}, "
        f"self_intersecting={result.self_intersecting}, "
        f"volume={result.volume}, "
        f"iota_guess={iota}, "
        f"iota_solved={result.solved_iota}"
    )
    if result.error_type is not None:
        print(
            "Boozer initialization raised "
            f"{result.error_type}: {result.error_message}"
        )
    raise RuntimeError("Something went wrong with the Boozer solve...")


def _attempt_shell(
    surf_prev,
    *,
    mpol,
    ntor,
    bs,
    vol_target,
    constraint_weight,
    iota,
    G,
    boozer_I,
    initial_surface_guess,
    nfp,
):
    """Non-raising single-shell solve used by the truncation branch."""
    return attempt_initialize_boozer_surface(
        surf_prev,
        mpol,
        ntor,
        bs,
        vol_target,
        constraint_weight,
        iota,
        G,
        boozer_I,
        initial_surface_guess=initial_surface_guess,
        nfp=nfp,
    )


def _accept_truncation_shell(result, previous_boozer_surface):
    """True iff a truncation shell solved, nests inside, and stays volume-ordered.

    ``result`` is the candidate ``BoozerInitializationResult`` for the next inner
    shell; ``previous_boozer_surface`` is the last accepted (larger) Boozer
    surface. Acceptance requires the solve to have succeeded, the new cross
    sections to nest strictly inside the previous one, and the new volume to be
    strictly smaller (inner-to-outer ordering, i.e. shrinking inward).
    """
    if not result.success:
        return False
    candidate_surface = result.boozer_surface.surface
    previous_surface = previous_boozer_surface.surface
    nested, _ = cross_sections_are_nested(candidate_surface, previous_surface)
    if not nested:
        return False
    return float(candidate_surface.volume()) < float(previous_surface.volume())


def _require_relaxed_family_postconditions(ordered_surface_data):
    """Relaxed nesting + volume-order check for a (possibly) truncated band.

    The truncated band is an outer suffix of the requested stack, so its names
    are not inner0-based and the strict name postcondition cannot run. This
    asserts only what holds for any accepted band: every adjacent pair nests
    (inner inside outer) and volumes are strictly increasing inner-to-outer.
    """
    volumes = [
        float(entry["boozer_surface"].surface.volume())
        for entry in ordered_surface_data
    ]
    if len(volumes) > 1 and not np.all(np.diff(volumes) > 0.0):
        names = [entry["name"] for entry in ordered_surface_data]
        raise RuntimeError(
            "boozer_surface_family accepted band volumes must be strictly "
            f"ordered inner-to-outer: {dict(zip(names, volumes))}."
        )
    for inner_entry, outer_entry in zip(
        ordered_surface_data[:-1], ordered_surface_data[1:]
    ):
        nested, bad_phis = cross_sections_are_nested(
            inner_entry["boozer_surface"].surface,
            outer_entry["boozer_surface"].surface,
        )
        if not nested:
            raise RuntimeError(
                "boozer_surface_family accepted band must nest "
                f"{inner_entry['name']} inside {outer_entry['name']}; "
                f"non-nested at phi={bad_phis}."
            )


def build_boozer_surface_family(
    ordered_configs,
    *,
    mpol,
    ntor,
    bs,
    constraint_weight,
    boozer_I,
    nfp,
    outer_seed,
    contract,
    make_entry,
    allow_truncation=False,
    min_surfaces=3,
):
    """Solve a nested Boozer-surface family outer-to-inner, returned inner-to-outer.

    ``ordered_configs``: inner-to-outer config dicts (each with ``name`` and
        ``target_volume``); the outer shell is ``ordered_configs[-1]``.
    ``outer_seed``: ``BoozerFamilyOuterSeed`` warm start for the outer shell
        (caller-validated surface/iota/G).
    ``contract``: ``contract_surface_to_target_volume`` -- contracts the previous
        solved surface toward the next inner target volume.
    ``make_entry``: ``_surface_data_entry`` -- builds the surface-data dict.
    ``allow_truncation``: when False (default) every shell must solve or the call
        raises (byte-identical to the driver's strict continuation); when True the
        march stops at the first shell that fails to solve / nest / stay ordered.
    ``min_surfaces``: minimum accepted shells; fewer than this raises.

    Returns ``(ordered_surface_data, provenance)`` with ``ordered_surface_data``
    the accepted band re-ordered inner-to-outer and ``provenance`` a
    ``BoozerFamilyProvenance``. In strict mode all requested shells are accepted,
    the relaxed postcondition is skipped (the caller runs the strict one), and
    ``provenance.truncated`` is False.
    """
    outer_config = ordered_configs[-1]
    outer_boozer_surface = _solve_or_raise(
        outer_seed.surface,
        mpol=mpol,
        ntor=ntor,
        bs=bs,
        vol_target=outer_config["target_volume"],
        constraint_weight=constraint_weight,
        iota=outer_seed.iota,
        G=outer_seed.G,
        boozer_I=boozer_I,
        initial_surface_guess=outer_seed.surface,
        nfp=nfp,
    )
    solved_by_name = {
        outer_config["name"]: make_entry(
            outer_config,
            outer_boozer_surface,
            "stage2_outer_seed",
        )
    }
    accepted_names = [outer_config["name"]]
    previous_name = outer_config["name"]
    previous_boozer_surface = outer_boozer_surface

    # March inner shells from outermost (innerN-2) to innermost (inner0).
    for config in reversed(ordered_configs[:-1]):
        surface_name = config["name"]
        if constraint_weight is None:
            # Exact-Newton build (constraint_weight=None): seed each inner shell from
            # the previous CONVERGED Boozer surface, UNcontracted. exact-Newton from a
            # contracted naive-scaled guess diverges (the documented reason 'ls' is the
            # default build); from the converged neighbor it root-finds the nearest
            # smaller-volume Boozer surface -- matching the offline nesting probe that
            # proved the stack nests at vol 0.045. The 'ls' build is unchanged below.
            continuation_surface = previous_boozer_surface.surface
        else:
            continuation_surface = contract(
                previous_boozer_surface.surface,
                config["target_volume"],
            )
        iota_guess = float(previous_boozer_surface.res["iota"])
        # Preserve the strict continuation's fail-fast on a non-finite warm-start G
        # (the driver wrapped this in ``_require_finite_explicit_G``). A solved
        # neighbor always carries a finite G, so this only fires on a degenerate
        # upstream solve -- but dropping it would defer that abort to an opaque
        # solver failure instead of a clear, attributable error.
        warm_start_G = previous_boozer_surface.res["G"]
        if warm_start_G is None or not np.isfinite(float(warm_start_G)):
            raise RuntimeError(
                "boozer_surface_family requires a finite warm-start G from "
                f"{previous_name} for {surface_name}; got {warm_start_G!r}."
            )
        G_guess = float(warm_start_G)

        if allow_truncation:
            result = _attempt_shell(
                continuation_surface,
                mpol=mpol,
                ntor=ntor,
                bs=bs,
                vol_target=config["target_volume"],
                constraint_weight=constraint_weight,
                iota=iota_guess,
                G=G_guess,
                boozer_I=boozer_I,
                initial_surface_guess=continuation_surface,
                nfp=nfp,
            )
            if not _accept_truncation_shell(result, previous_boozer_surface):
                break
            boozer_surface = result.boozer_surface
        else:
            boozer_surface = _solve_or_raise(
                continuation_surface,
                mpol=mpol,
                ntor=ntor,
                bs=bs,
                vol_target=config["target_volume"],
                constraint_weight=constraint_weight,
                iota=iota_guess,
                G=G_guess,
                boozer_I=boozer_I,
                initial_surface_guess=continuation_surface,
                nfp=nfp,
            )

        solved_by_name[surface_name] = make_entry(
            config,
            boozer_surface,
            f"{previous_name}_continuation_{surface_name}",
        )
        accepted_names.append(surface_name)
        previous_name = surface_name
        previous_boozer_surface = boozer_surface

    if len(accepted_names) < min_surfaces:
        raise RuntimeError(
            "boozer_surface_family accepted "
            f"{len(accepted_names)} surface(s) "
            f"({accepted_names}) but requires at least {min_surfaces}."
        )

    # Re-order the accepted band inner-to-outer: it is an outer suffix of the
    # requested stack, so order by the requested-config sequence and keep only
    # accepted names.
    accepted = set(accepted_names)
    ordered_surface_data = [
        solved_by_name[config["name"]]
        for config in ordered_configs
        if config["name"] in accepted
    ]

    truncated = len(ordered_surface_data) < len(ordered_configs)
    if truncated:
        _require_relaxed_family_postconditions(ordered_surface_data)

    provenance = BoozerFamilyProvenance(
        accepted=len(ordered_surface_data),
        requested=len(ordered_configs),
        last_good_name=previous_name,
        truncated=truncated,
    )
    return ordered_surface_data, provenance
