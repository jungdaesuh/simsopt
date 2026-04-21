"""Shared plotting utilities for the single-stage optimization workflow.

Extracted from banana_coil_solver.py and single_stage_banana_example.py
to eliminate code duplication (Issue #23).
"""
import numpy as np
import matplotlib.pyplot as plt


def norm_field_plot(surf, bs, filename):
    """Plot the relative normal magnetic field B·n/|B| on a surface.

    Computes the area-weighted mean |B·n|/|B| and generates a contourf plot.

    Args:
        surf: Surface object with gamma(), normal(), quadpoints_phi/theta.
        bs: BiotSavart field object.
        filename: Output path (without extension; .png is appended).

    Returns:
        mean_abs_relBfinal_norm: Area-weighted mean of |B·n|/|B|.
    """
    theta = surf.quadpoints_theta
    phi = surf.quadpoints_phi
    n = surf.normal()
    absn = np.linalg.norm(n, axis=2)
    unitn = n * (1. / absn)[:, :, None]
    sqrt_area = np.sqrt(absn.reshape((-1, 1)) / float(absn.size))
    surf_area = sqrt_area**2
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bfinal = bs.B().reshape(n.shape)
    Bfinal_norm = np.sum(Bfinal * unitn, axis=2)[:, :, None]
    modBfinal = np.sqrt(np.sum(Bfinal**2, axis=2))[:, :, None]
    relBfinal_norm = Bfinal_norm / modBfinal
    abs_relBfinal_norm_dA = np.abs(relBfinal_norm.reshape((-1, 1))) * surf_area
    mean_abs_relBfinal_norm = np.sum(abs_relBfinal_norm_dA) / np.sum(surf_area)
    max_rnorm = np.max(np.abs(relBfinal_norm))

    fig, ax = plt.subplots()
    contour = ax.contourf(phi, theta, np.squeeze(relBfinal_norm).T, levels=50,
                          cmap='seismic', vmin=-max_rnorm, vmax=max_rnorm)
    ax.set_xlabel(r'$\phi/2\pi$', fontsize=18, fontweight='bold')
    ax.set_ylabel(r'$\theta/2\pi$', fontsize=18, fontweight='bold')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.set_ylabel(r'$\mathbf{B}\cdot\mathbf{n}/|\mathbf{B}|$', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(axis='y', which='major', labelsize=14)
    ax.set_title(f'Surface-averaged |Bn|/|B| = {mean_abs_relBfinal_norm:.4e}', fontsize=18, fontweight='bold')
    plt.savefig(f"{filename}.png")
    plt.close()

    return mean_abs_relBfinal_norm, modBfinal, surf_area, phi, theta


def magnitude_field_plot(modBfinal, surf_area, phi, theta, filename):
    """Plot the magnetic field magnitude |B| on a surface.

    Args:
        modBfinal: Field magnitude array (nphi, ntheta, 1).
        surf_area: Area element weights.
        phi: Toroidal quadrature points.
        theta: Poloidal quadrature points.
        filename: Output path (without extension; .png is appended).
    """
    abs_modBfinal_dA = np.abs(modBfinal.reshape((-1, 1))) * surf_area
    mean_abs_modBfinal = np.sum(abs_modBfinal_dA) / np.sum(surf_area)
    fig, ax = plt.subplots()
    contour = ax.contour(phi, theta, np.squeeze(modBfinal).T, levels=25, cmap='viridis')
    ax.set_xlabel(r'$\phi/2\pi$', fontsize=18, fontweight='bold')
    ax.set_ylabel(r'$\theta/2\pi$', fontsize=18, fontweight='bold')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.set_ylabel(r'$|\mathbf{B}|$', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(axis='y', which='major', labelsize=14)
    ax.set_title(f'Surface-averaged |B| = {mean_abs_modBfinal:.3f}', fontsize=18, fontweight='bold')
    plt.savefig(f"{filename}.png")
    plt.close()


def _closed_rz(cross_section):
    """Convert a Cartesian cross-section to closed (R, Z) arrays."""
    r = np.sqrt(cross_section[:, 0]**2 + cross_section[:, 1]**2)
    z = cross_section[:, 2]
    return np.append(r, r[0]), np.append(z, z[0])


def cross_section_plot(
    surf_coils,
    surf,
    banana_curve,
    filename,
    lcfs_clearance_reference,
    VV,
):
    """Plot toroidal cross-sections of the plasma, coil surface, and vessel.

    Args:
        surf_coils: Coil winding surface.
        surf: Plasma surface.
        banana_curve: Banana coil curve (plotted as R-Z projection).
        filename: Output path (without extension; .png is appended).
        lcfs_clearance_reference: Concentric LCFS clearance-reference surface.
        VV: Vacuum vessel surface.
    """
    plt.figure(figsize=(7, 6))
    # Banana coil R-Z projection (open curve, not a cross-section)
    gamma = banana_curve.gamma()
    plt.plot(np.sqrt(gamma[:, 0]**2 + gamma[:, 1]**2), gamma[:, 2],
             'k--', linewidth=1.5, label='Banana Coil')
    # Fixed geometry cross-sections at phi=0
    for surface, label in [
        (surf_coils, "Banana Surface"),
        (lcfs_clearance_reference, "LCFS Clearance Ref"),
        (VV, "Vacuum Vessel"),
    ]:
        r, z = _closed_rz(surface.cross_section(0))
        plt.plot(r, z, label=label)
    # Plasma cross-sections at multiple toroidal angles
    phi_array = np.linspace(0, 2 * np.pi / surf_coils.nfp * 4 / 5, 5)
    for phi_slice in phi_array:
        r, z = _closed_rz(surf.cross_section(phi_slice / (2 * np.pi)))
        plt.plot(r, z, label=f'\u03a6={phi_slice/np.pi:0.2f}\u03c0')
    plt.xlabel('R [m]', fontsize=18, fontweight='bold')
    plt.ylabel('Z [m]', fontsize=18, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.minorticks_on()
    plt.grid(True)
    plt.savefig(f"{filename}.png")
    plt.close()
