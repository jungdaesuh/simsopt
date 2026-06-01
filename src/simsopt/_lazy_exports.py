import ast
import importlib
from collections.abc import Mapping
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


def build_lazy_export_map(
    package_file: str,
    module_names: tuple[str, ...],
    *,
    package_exports_by_module: Mapping[str, tuple[str, ...]] | None = None,
    package_export_order: tuple[str, ...] | None = None,
):
    package_dir = Path(package_file).resolve().parent
    export_to_module = {}
    export_order = []
    for module_name in module_names:
        if (
            package_exports_by_module is not None
            and module_name in package_exports_by_module
        ):
            module_exports = package_exports_by_module[module_name]
        else:
            module_path = package_dir / f"{module_name}.py"
            module_exports = _module_exports(str(module_path))
        for export_name in module_exports:
            previous_module = export_to_module.get(export_name)
            if previous_module is not None:
                raise RuntimeError(
                    f"duplicate package export {export_name!r}: "
                    f"{previous_module!r} and {module_name!r}"
                )
            export_to_module[export_name] = module_name
            export_order.append(export_name)
    if package_export_order is None:
        return export_to_module, tuple(export_order)
    package_export_names = set()
    duplicate_order_names = set()
    for name in package_export_order:
        if name in package_export_names:
            duplicate_order_names.add(name)
        package_export_names.add(name)
    if duplicate_order_names:
        raise RuntimeError(
            f"duplicate package export order entries: {sorted(duplicate_order_names)}"
        )
    missing_order_names = [
        name for name in package_export_order if name not in export_to_module
    ]
    if missing_order_names:
        raise RuntimeError(
            f"package export order references unknown exports: {missing_order_names}"
        )
    extra_export_names = [
        name for name in export_order if name not in package_export_names
    ]
    if extra_export_names:
        raise RuntimeError(
            f"package export order omits module exports: {extra_export_names}"
        )
    ordered_export_to_module = {
        export_name: export_to_module[export_name]
        for export_name in package_export_order
    }
    return ordered_export_to_module, package_export_order


def resolve_lazy_export(package_name: str, export_to_module, name: str):
    module_name = export_to_module.get(name)
    if module_name is None:
        raise AttributeError(f"module {package_name!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", package_name)
    return getattr(module, name)
