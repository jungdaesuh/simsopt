import pytest

from simsopt._lazy_exports import build_lazy_export_map


@pytest.mark.parametrize("package_export_order", [None, ("x",)])
def test_build_lazy_export_map_rejects_same_module_duplicate_exports(
    tmp_path,
    package_export_order,
):
    package_file = tmp_path / "__init__.py"
    package_file.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="duplicate package export 'x'"):
        build_lazy_export_map(
            str(package_file),
            ("module",),
            package_exports_by_module={"module": ("x", "x")},
            package_export_order=package_export_order,
        )
