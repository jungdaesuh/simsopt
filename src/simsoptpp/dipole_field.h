#pragma once

#include <cmath>
#include <tuple>  // c++ tuples
#include <string> // for string class
#include <iostream>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

// Raw point-dipole kernels. They intentionally do not regularize singular
// inputs: evaluation points that coincide with dipole locations can produce
// non-finite values. Callers must keep points off dipole locations unless they
// are explicitly testing the singular contract. Arrays are double-precision
// pyarrays, so Python float32 inputs are widened to float64 by the binding.
Array dipole_field_B(Array& points, Array& m_points, Array& m);

Array dipole_field_A(Array& points, Array& m_points, Array& m);

Array dipole_field_dB(Array& points, Array& m_points, Array& m);

Array dipole_field_dA(Array& points, Array& m_points, Array& m);

// For cylindrical or toroidal coordinate_flag, degenerate basis angles follow
// the std::atan2 zero-angle convention to match the pure-JAX implementation.
Array dipole_field_Bn(Array& points, Array& m_points, Array& unitnormal, int nfp, int stellsym, Array& b, std::string coordinate_flag="cartesian", double R0=0.0);

Array define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(Array& normal_inner, Array& normal_outer, Array& xyz_uniform, Array& xyz_inner, Array& xyz_outer);
