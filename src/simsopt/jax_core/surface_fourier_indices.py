"""Index helpers for Fourier surface coefficient layouts."""

import numpy as np

__all__ = ["stellsym_scatter_indices"]


def _is_stellsym_xy(m: int, n: int, mpol: int, ntor: int) -> bool:
    """True if ``(m, n)`` is free for x under stellarator symmetry."""
    is_cos_theta = m <= mpol
    is_sin_theta = m > mpol
    is_cos_phi = n <= ntor
    is_sin_phi = n > ntor
    return (is_cos_theta and is_cos_phi) or (is_sin_theta and is_sin_phi)


def _is_stellsym_z(m: int, n: int, mpol: int, ntor: int) -> bool:
    """True if ``(m, n)`` is free for y/z under stellarator symmetry."""
    is_cos_theta = m <= mpol
    is_sin_theta = m > mpol
    is_cos_phi = n <= ntor
    is_sin_phi = n > ntor
    return (is_cos_theta and is_sin_phi) or (is_sin_theta and is_cos_phi)


def stellsym_scatter_indices(mpol: int, ntor: int) -> np.ndarray:
    """Compute scatter indices for stellsym DOF unpacking.

    The returned array maps each DOF index to its flattened position in the
    ``[xc, yc, zc]`` coefficient super-vector.
    """
    n_per_coord = (2 * mpol + 1) * (2 * ntor + 1)
    indices: list[int] = []
    for coord_offset, allowed_fn in (
        (0, _is_stellsym_xy),
        (n_per_coord, _is_stellsym_z),
        (2 * n_per_coord, _is_stellsym_z),
    ):
        for m in range(2 * mpol + 1):
            for n in range(2 * ntor + 1):
                if allowed_fn(m, n, mpol, ntor):
                    indices.append(coord_offset + m * (2 * ntor + 1) + n)
    return np.asarray(indices, dtype=np.int32)
