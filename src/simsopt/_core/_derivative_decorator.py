"""Cycle-free owner for the public derivative decorator."""

from functools import wraps
from typing import Callable, Protocol, TypeVar


class _ProjectedDerivative(Protocol):
    """Callable projection contract returned by derivative implementations."""

    def __call__(self, optim: object) -> object: ...


_DerivativeResult = TypeVar("_DerivativeResult", bound=_ProjectedDerivative)


def derivative_dec(func: Callable[..., _DerivativeResult]) -> Callable[..., object]:
    """Project a derivative result unless the caller requests its partials."""

    @wraps(func)
    def _derivative_dec(
        self: object,
        *args: object,
        partials: bool = False,
        **kwargs: object,
    ) -> object:
        derivative = func(self, *args, **kwargs)
        if partials:
            return derivative
        return derivative(self)

    return _derivative_dec
