_OPTIMIZABLE_CONTRACT_ATTRS = (
    "unique_dof_lineage",
    "local_full_dof_size",
    "local_dof_size",
    "x",
    "dof_size",
)


def has_optimizable_contract(value):
    return all(hasattr(value, attr) for attr in _OPTIMIZABLE_CONTRACT_ATTRS)
