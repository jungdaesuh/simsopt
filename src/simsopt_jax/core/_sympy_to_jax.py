"""Static SymPy expression lowering for small JAX kernel builders."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import sympy as sp
from sympy.core.relational import Equality, GreaterThan, Relational
from sympy.core.relational import StrictGreaterThan, StrictLessThan, Unequality
from sympy.core.relational import LessThan


ExpressionEvaluator = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
ConditionEvaluator = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
VectorEvaluator = Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]

_R_SYMBOL, _Z_SYMBOL, _PHI_SYMBOL = sp.symbols("R Z phi")
_SUPPORTED_SYMBOLS = frozenset((_R_SYMBOL, _Z_SYMBOL, _PHI_SYMBOL))

_FUNCTIONS: dict[object, Callable[..., jax.Array]] = {
    sp.sin: jnp.sin,
    sp.cos: jnp.cos,
    sp.tan: jnp.tan,
    sp.asin: jnp.arcsin,
    sp.acos: jnp.arccos,
    sp.atan: jnp.arctan,
    sp.atan2: jnp.arctan2,
    sp.sinh: jnp.sinh,
    sp.cosh: jnp.cosh,
    sp.tanh: jnp.tanh,
    sp.asinh: jnp.arcsinh,
    sp.acosh: jnp.arccosh,
    sp.atanh: jnp.arctanh,
    sp.exp: jnp.exp,
    sp.log: jnp.log,
    sp.sec: lambda x: 1.0 / jnp.cos(x),
    sp.csc: lambda x: 1.0 / jnp.sin(x),
    sp.cot: lambda x: 1.0 / jnp.tan(x),
    sp.sqrt: jnp.sqrt,
}

_RELATIONAL_FUNCTIONS: dict[
    type[Relational], Callable[[jax.Array, jax.Array], jax.Array]
] = {
    StrictGreaterThan: jnp.greater,
    GreaterThan: jnp.greater_equal,
    StrictLessThan: jnp.less,
    LessThan: jnp.less_equal,
    Equality: jnp.equal,
    Unequality: jnp.not_equal,
}


def lower_sympy_expression(expr: sp.Expr) -> ExpressionEvaluator:
    """Return a pure JAX evaluator for a supported SymPy expression."""

    _validate_supported(expr)

    def evaluate(R: jax.Array, Z: jax.Array, Phi: jax.Array) -> jax.Array:
        return _eval_expr(expr, R, Z, Phi)

    return evaluate


def _validate_supported(expr: sp.Expr) -> None:
    unsupported = tuple(sorted(expr.free_symbols - _SUPPORTED_SYMBOLS, key=str))
    if unsupported:
        raise NotImplementedError(
            "Unsupported symbols in scalar-potential expression: "
            + ", ".join(str(symbol) for symbol in unsupported)
        )
    if expr in _SUPPORTED_SYMBOLS:
        return
    if expr.is_number:
        if expr.is_real is not True:
            raise NotImplementedError(
                "Unsupported complex numeric constant in scalar-potential "
                f"expression: {expr!s}"
            )
        return
    if isinstance(expr, sp.Piecewise):
        if not any(condition == sp.S.true for _, condition in expr.args):
            raise NotImplementedError(
                "Unsupported Piecewise scalar-potential expression without "
                "a True default branch."
            )
        for value, condition in expr.args:
            _validate_supported(value)
            _validate_condition(condition)
        return
    if isinstance(expr, (sp.Add, sp.Mul, sp.Pow)) or expr.func in _FUNCTIONS:
        for arg in expr.args:
            _validate_supported(arg)
        return
    raise NotImplementedError(
        f"Unsupported SymPy node in scalar-potential expression: {expr.func!s}"
    )


def _validate_condition(condition: sp.Expr) -> None:
    if condition in (sp.S.true, sp.S.false):
        return
    if isinstance(condition, Relational):
        if type(condition) not in _RELATIONAL_FUNCTIONS:
            raise NotImplementedError(
                "Unsupported relational condition in scalar-potential "
                f"Piecewise expression: {condition.func!s}"
            )
        _validate_supported(condition.lhs)
        _validate_supported(condition.rhs)
        return
    if condition.func in (sp.And, sp.Or):
        for arg in condition.args:
            _validate_condition(arg)
        return
    if condition.func is sp.Not:
        _validate_condition(condition.args[0])
        return
    raise NotImplementedError(
        "Unsupported Boolean condition in scalar-potential Piecewise "
        f"expression: {condition.func!s}"
    )


def lower_sympy_expressions(exprs: Sequence[sp.Expr]) -> VectorEvaluator:
    """Return a pure JAX evaluator for a fixed list of SymPy expressions."""

    evaluators = tuple(lower_sympy_expression(expr) for expr in exprs)

    def evaluate(R: jax.Array, Z: jax.Array, Phi: jax.Array) -> tuple[jax.Array, ...]:
        return tuple(evaluator(R, Z, Phi) for evaluator in evaluators)

    return evaluate


def _constant_like(value: sp.Expr, reference: jax.Array) -> jax.Array:
    return jnp.asarray(float(value), dtype=reference.dtype) + jnp.zeros_like(reference)


def _eval_expr(expr: sp.Expr, R: jax.Array, Z: jax.Array, Phi: jax.Array) -> jax.Array:
    if expr == _R_SYMBOL:
        return R
    if expr == _Z_SYMBOL:
        return Z
    if expr == _PHI_SYMBOL:
        return Phi
    if expr.is_number:
        return _constant_like(expr, R)
    if isinstance(expr, sp.Add):
        total = jnp.zeros_like(R)
        for arg in expr.args:
            total = total + _eval_expr(arg, R, Z, Phi)
        return total
    if isinstance(expr, sp.Mul):
        product = jnp.ones_like(R)
        for arg in expr.args:
            product = product * _eval_expr(arg, R, Z, Phi)
        return product
    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        return jnp.power(_eval_expr(base, R, Z, Phi), _eval_expr(exponent, R, Z, Phi))
    if isinstance(expr, sp.Piecewise):
        result = jnp.zeros_like(R)
        for value, condition in reversed(expr.args):
            result = jnp.where(
                _eval_condition(condition, R, Z, Phi),
                _eval_expr(value, R, Z, Phi),
                result,
            )
        return result

    func = _FUNCTIONS.get(expr.func)
    if func is not None:
        return func(*(_eval_expr(arg, R, Z, Phi) for arg in expr.args))

    raise NotImplementedError(
        f"Unsupported SymPy node in scalar-potential expression: {expr.func!s}"
    )


def _eval_condition(
    condition: sp.Expr, R: jax.Array, Z: jax.Array, Phi: jax.Array
) -> jax.Array:
    if condition == sp.S.true:
        return jnp.ones_like(R, dtype=bool)
    if condition == sp.S.false:
        return jnp.zeros_like(R, dtype=bool)
    if isinstance(condition, Relational):
        func = _RELATIONAL_FUNCTIONS.get(type(condition))
        if func is not None:
            return func(
                _eval_expr(condition.lhs, R, Z, Phi),
                _eval_expr(condition.rhs, R, Z, Phi),
            )
    if condition.func is sp.And:
        result = jnp.ones_like(R, dtype=bool)
        for arg in condition.args:
            result = result & _eval_condition(arg, R, Z, Phi)
        return result
    if condition.func is sp.Or:
        result = jnp.zeros_like(R, dtype=bool)
        for arg in condition.args:
            result = result | _eval_condition(arg, R, Z, Phi)
        return result
    if condition.func is sp.Not:
        return ~_eval_condition(condition.args[0], R, Z, Phi)
    raise NotImplementedError(
        "Unsupported Boolean condition in scalar-potential Piecewise "
        f"expression: {condition.func!s}"
    )
