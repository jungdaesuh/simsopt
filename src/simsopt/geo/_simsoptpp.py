import types

try:
    import simsoptpp as _SIMSOPTPP
except ImportError:
    _SIMSOPTPP = None

_PLACEHOLDER_BASES = {}
_PLACEHOLDER_FUNCTIONS = {}


def has_simsoptpp_symbol(symbol_name):
    return _SIMSOPTPP is not None and hasattr(_SIMSOPTPP, symbol_name)


def _raise_missing_simsoptpp(symbol_name, *, action):
    raise ImportError(f"simsoptpp is required to {action} {symbol_name}")


def _placeholder_base(symbol_name):
    base = _PLACEHOLDER_BASES.get(symbol_name)
    if base is None:

        def __new__(cls, *_args, **_kwargs):
            _raise_missing_simsoptpp(cls.__name__, action="instantiate")

        def __init__(self, *_args, **_kwargs):
            _raise_missing_simsoptpp(type(self).__name__, action="instantiate")

        base = type(
            f"_MissingSimsoptpp{symbol_name}",
            (),
            {"__new__": __new__, "__init__": __init__},
        )
        _PLACEHOLDER_BASES[symbol_name] = base
    return base


def _placeholder_function(symbol_name):
    fn = _PLACEHOLDER_FUNCTIONS.get(symbol_name)
    if fn is None:

        def _missing(*_args, **_kwargs):
            _raise_missing_simsoptpp(symbol_name, action="use")

        fn = _missing
        _PLACEHOLDER_FUNCTIONS[symbol_name] = fn
    return fn


def _normalize_symbol_names(symbol_names):
    if len(symbol_names) == 1 and isinstance(symbol_names[0], (list, tuple)):
        return tuple(symbol_names[0])
    return tuple(symbol_names)


def _resolve_optional_symbol(symbol_name, *, placeholder):
    if has_simsoptpp_symbol(symbol_name):
        return getattr(_SIMSOPTPP, symbol_name)
    return placeholder(symbol_name)


def sopp_namespace(*symbol_names, kind="class"):
    symbol_names = _normalize_symbol_names(symbol_names)
    if not symbol_names:
        raise ValueError("at least one simsoptpp symbol name is required")
    if (
        len(symbol_names) == 1
        and kind == "class"
        and has_simsoptpp_symbol(symbol_names[0])
    ):
        return _SIMSOPTPP
    if kind == "class":
        placeholder = _placeholder_base
    elif kind == "function":
        placeholder = _placeholder_function
    else:
        raise ValueError(f"unsupported simsoptpp symbol kind: {kind}")
    resolved = {
        symbol_name: _resolve_optional_symbol(symbol_name, placeholder=placeholder)
        for symbol_name in symbol_names
    }
    return types.SimpleNamespace(**resolved)
