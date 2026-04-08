import ast
import importlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def _module_exports(module_path: str) -> tuple[str, ...]:
    path = Path(module_path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                exports = ast.literal_eval(node.value)
                if not isinstance(exports, (list, tuple)):
                    raise TypeError(f"{path} __all__ must be a list or tuple literal")
                return tuple(str(name) for name in exports)
    raise RuntimeError(f"{path} does not define a literal __all__")


def build_lazy_export_map(package_file: str, module_names: tuple[str, ...]):
    package_dir = Path(package_file).resolve().parent
    export_to_module = {}
    export_order = []
    for module_name in module_names:
        module_path = package_dir / f"{module_name}.py"
        for export_name in _module_exports(str(module_path)):
            previous_module = export_to_module.get(export_name)
            if previous_module is not None and previous_module != module_name:
                raise RuntimeError(
                    f"duplicate package export {export_name!r}: "
                    f"{previous_module!r} and {module_name!r}"
                )
            export_to_module[export_name] = module_name
            export_order.append(export_name)
    return export_to_module, tuple(export_order)


def resolve_lazy_export(package_name: str, export_to_module, name: str):
    module_name = export_to_module.get(name)
    if module_name is None:
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", package_name)
    return getattr(module, name)
