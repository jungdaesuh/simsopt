import types

try:
    import simsoptpp as _SIMSOPTPP
except ImportError:
    _SIMSOPTPP = None

_PLACEHOLDER_BASES = {}


def has_simsoptpp_symbol(symbol_name):
    return _SIMSOPTPP is not None and hasattr(_SIMSOPTPP, symbol_name)


def _raise_missing_simsoptpp(symbol_name):
    raise ImportError(f"simsoptpp is required to instantiate {symbol_name}")


def _placeholder_base(symbol_name):
    base = _PLACEHOLDER_BASES.get(symbol_name)
    if base is None:

        def __new__(cls, *_args, **_kwargs):
            _raise_missing_simsoptpp(cls.__name__)

        def __init__(self, *_args, **_kwargs):
            _raise_missing_simsoptpp(type(self).__name__)

        base = type(
            f"_MissingSimsoptpp{symbol_name}",
            (),
            {"__new__": __new__, "__init__": __init__},
        )
        _PLACEHOLDER_BASES[symbol_name] = base
    return base


def sopp_namespace(symbol_name):
    if has_simsoptpp_symbol(symbol_name):
        return _SIMSOPTPP
    return types.SimpleNamespace(**{symbol_name: _placeholder_base(symbol_name)})
