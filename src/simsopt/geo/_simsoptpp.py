import types

try:
    import simsoptpp as _SIMSOPTPP
except ImportError:
    _SIMSOPTPP = None

_PLACEHOLDER_BASES = {}


def has_simsoptpp_symbol(symbol_name):
    return _SIMSOPTPP is not None and hasattr(_SIMSOPTPP, symbol_name)


def _placeholder_base(symbol_name):
    base = _PLACEHOLDER_BASES.get(symbol_name)
    if base is None:
        base = type(f"_MissingSimsoptpp{symbol_name}", (), {})
        _PLACEHOLDER_BASES[symbol_name] = base
    return base


def sopp_namespace(symbol_name):
    if has_simsoptpp_symbol(symbol_name):
        return _SIMSOPTPP
    return types.SimpleNamespace(**{symbol_name: _placeholder_base(symbol_name)})
