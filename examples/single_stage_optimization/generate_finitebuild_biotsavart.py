from __future__ import annotations

from import_provenance import configure_local_simsopt_imports

configure_local_simsopt_imports(__file__)

from banana_opt.finitebuild_export import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
