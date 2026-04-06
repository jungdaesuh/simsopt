_OPTIMIZABLE_CONTRACT_ATTRS = (
    "unique_dof_lineage",
    "local_full_dof_size",
    "local_dof_size",
    "x",
    "dof_size",
)

_DERIVATIVE_TARGET_CONTRACT_ATTRS = ("unique_dof_lineage",)
_DERIVATIVE_KEY_CONTRACT_ATTRS = ("local_full_dof_size",)


def has_optimizable_contract(value):
    return all(hasattr(value, attr) for attr in _OPTIMIZABLE_CONTRACT_ATTRS)


def has_derivative_target_contract(value):
    return all(hasattr(value, attr) for attr in _DERIVATIVE_TARGET_CONTRACT_ATTRS)


def has_derivative_key_contract(value):
    return all(hasattr(value, attr) for attr in _DERIVATIVE_KEY_CONTRACT_ATTRS)
