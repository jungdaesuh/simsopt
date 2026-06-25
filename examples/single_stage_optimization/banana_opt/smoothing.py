import numpy as np


def stable_softmax(values, smoothing_eps: float):
    shifted = np.asarray(values, dtype=float) - float(np.max(values))
    weights = np.exp(shifted)
    total = max(float(np.sum(weights)), float(smoothing_eps))
    return weights / total


def smoothmax_selected(values, temperature: float, smoothing_eps: float):
    bounded_temperature = max(float(temperature), float(smoothing_eps))
    values_array = np.asarray(values, dtype=float)
    shifted = (values_array - float(np.max(values_array))) / bounded_temperature
    weights = stable_softmax(shifted, smoothing_eps)
    smooth_value = float(np.max(values_array)) + bounded_temperature * float(
        np.log(np.sum(np.exp(shifted)))
    )
    return smooth_value, weights


def smoothmin_selected(values, temperature: float, smoothing_eps: float):
    bounded_temperature = max(float(temperature), float(smoothing_eps))
    values_array = np.asarray(values, dtype=float)
    minimum_value = float(np.min(values_array))
    shifted = -(values_array - minimum_value) / bounded_temperature
    weights = stable_softmax(shifted, smoothing_eps)
    smooth_value = minimum_value - bounded_temperature * float(
        np.log(np.sum(np.exp(shifted)))
    )
    return smooth_value, weights
