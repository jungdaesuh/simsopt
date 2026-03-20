#pragma once

#include <algorithm>
#include <stdexcept>

namespace simsoptpp {

// pybind11 2.13 adds in-place operators on the Python object base type, which
// makes whole-array *= ambiguous for xt::pyarray. Use iterator-based updates so
// the same templated code works for both native xtensor containers and pyarray.
template<class Array, class Scalar>
inline void fill_array(Array& data, Scalar value) {
    using value_type = typename Array::value_type;
    const value_type typed_value = static_cast<value_type>(value);
    for (auto it = data.begin(); it != data.end(); ++it) {
        *it = typed_value;
    }
}

template<class Array, class Scalar>
inline void scale_array(Array& data, Scalar factor) {
    using value_type = typename Array::value_type;
    const value_type typed_factor = static_cast<value_type>(factor);
    for (auto it = data.begin(); it != data.end(); ++it) {
        *it = (*it) * typed_factor;
    }
}

template<class Array, class OtherArray, class Scalar>
inline void axpy_array(Array& dest, const OtherArray& src, Scalar factor) {
    if (dest.dimension() != src.dimension()
            || !std::equal(dest.shape().begin(), dest.shape().end(), src.shape().begin())) {
        throw std::invalid_argument("simsoptpp axpy_array shape mismatch");
    }
    using value_type = typename Array::value_type;
    const value_type typed_factor = static_cast<value_type>(factor);
    auto dest_it = dest.begin();
    auto src_it = src.begin();
    for (; dest_it != dest.end(); ++dest_it, ++src_it) {
        *dest_it = *dest_it + (*src_it) * typed_factor;
    }
}

template<class Array, class OtherArray>
inline void subtract_array(Array& dest, const OtherArray& src) {
    if (dest.dimension() != src.dimension()
            || !std::equal(dest.shape().begin(), dest.shape().end(), src.shape().begin())) {
        throw std::invalid_argument("simsoptpp subtract_array shape mismatch");
    }
    auto dest_it = dest.begin();
    auto src_it = src.begin();
    for (; dest_it != dest.end(); ++dest_it, ++src_it) {
        *dest_it = *dest_it - *src_it;
    }
}

}  // namespace simsoptpp
