"""The nested-LS runtime identity block must name the executed C++ binary.

Every nested-LS evidence document embeds ``nested_ls_runtime_identity()``.
The native lane's numbers come out of the compiled ``simsoptpp`` extension,
and that binary has already been swapped mid-campaign (``41b2ca79…`` →
``95190afa…`` → ``d4a6e028…``), so an identity block that records the host,
JAX backend and threading env but not the binary cannot distinguish a real
result change from a rebuild.

These tests assert observed state: the digest of real bytes, the identity of
a really-mapped shared object, and a real JSON round trip.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import simsoptpp
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    dump_strict_json,
    nested_ls_receipt_provenance,
    nested_ls_runtime_identity,
)

_PROC_MAPS = Path("/proc/self/maps")


def _imported_extension_path() -> Path:
    """Resolve the extension path from the module object in ``sys.modules``.

    Derived here independently of the production helper so the assertions
    below compare two derivations rather than one derivation with itself.
    """

    module = sys.modules["simsoptpp"]
    assert module is simsoptpp
    file_attribute = module.__file__
    assert file_attribute is not None
    return Path(file_attribute).resolve()


def test_reported_sha256_is_the_digest_of_the_imported_extension_bytes() -> None:
    identity = nested_ls_runtime_identity()
    expected = hashlib.sha256(_imported_extension_path().read_bytes()).hexdigest()
    assert identity["simsoptpp_sha256"] == expected, (
        "the identity block's simsoptpp_sha256 is not the digest of the bytes "
        f"at {_imported_extension_path()}; recorded "
        f"{identity['simsoptpp_sha256']!r}, independently computed "
        f"{expected!r}."
    )


def test_reported_path_is_the_resolved_path_of_the_imported_module() -> None:
    identity = nested_ls_runtime_identity()
    reported = Path(str(identity["simsoptpp_path"]))
    expected = _imported_extension_path()
    assert reported == expected, (
        "simsoptpp_path does not name the module object this process "
        f"imported; recorded {reported}, sys.modules['simsoptpp'].__file__ "
        f"resolves to {expected}."
    )
    assert os.stat(reported).st_ino == os.stat(expected).st_ino, (
        "simsoptpp_path names a different inode than the imported module's "
        "file: the recorded path is a same-named copy, not the same file."
    )


_DELETED_MAPPING_SUFFIX = " (deleted)"


def _file_backed_mappings() -> tuple[frozenset[str], frozenset[str]]:
    """Split this process's file-backed mappings into live and replaced.

    A ``/proc/self/maps`` row is ``address perms offset dev inode pathname``,
    so the path is everything after the fifth field — it is NOT the last
    whitespace-separated token, both because a path may contain spaces and
    because Linux appends ``" (deleted)"`` to the row when the mapped file's
    directory entry has been removed. That suffix is the whole point here:
    swapping the extension on disk while it is loaded is the documented drift
    this identity field exists to record, and it renders as a deleted mapping
    rather than as no mapping at all.
    """

    live: set[str] = set()
    replaced: set[str] = set()
    for line in _PROC_MAPS.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5]
        if not path.startswith("/"):
            continue
        if path.endswith(_DELETED_MAPPING_SUFFIX):
            replaced.add(path[: -len(_DELETED_MAPPING_SUFFIX)])
        else:
            live.add(path)
    return frozenset(live), frozenset(replaced)


@pytest.mark.skipif(not _PROC_MAPS.exists(), reason="needs /proc/self/maps")
def test_reported_path_is_mapped_into_this_process() -> None:
    """A file that merely exists is not evidence; a mapped one is."""

    identity = nested_ls_runtime_identity()
    reported = str(identity["simsoptpp_path"])
    live, replaced = _file_backed_mappings()
    assert reported not in replaced, (
        f"simsoptpp_path {reported} IS mapped into this process, but the file "
        "at that path has been replaced since it was loaded (Linux marks the "
        "mapping '(deleted)'). The recorded simsoptpp_sha256 therefore digests "
        "bytes that are NOT the ones executing — the exact extension-swap "
        "drift this field exists to record. Re-import in a fresh process "
        "before minting evidence."
    )
    assert reported in live, (
        f"simsoptpp_path {reported} is not among the file-backed mappings of "
        "this process at all, so it is a path that exists rather than the "
        "binary that is executing."
    )


def test_reported_binding_follows_the_module_attribute_not_the_venv_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-point ``simsoptpp.__file__`` and both reported values must move.

    This is what distinguishes reading the imported module from guessing the
    path out of the venv layout or re-resolving the name with ``find_spec``:
    only the former tracks the attribute.
    """

    substitute = tmp_path / "simsoptpp.cpython-311-x86_64-linux-gnu.so"
    substitute.write_bytes(b"not the real extension\n")
    monkeypatch.setattr(simsoptpp, "__file__", str(substitute))

    identity = nested_ls_runtime_identity()
    assert Path(str(identity["simsoptpp_path"])) == substitute.resolve()
    assert (
        identity["simsoptpp_sha256"]
        == hashlib.sha256(b"not the real extension\n").hexdigest()
    ), (
        "the recorded digest did not follow simsoptpp.__file__, so the "
        "identity block is hashing a path derived some other way and would "
        "keep reporting a stale binary after an extension swap."
    )


def test_identity_refuses_to_report_an_unhashable_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An extension with no readable path fails the document, not blanks it.

    The ``__file__ = None`` state here is SYNTHETIC — it is monkeypatched, not
    a state any real build of ``simsoptpp`` reaches. (A truly statically
    linked extension would appear in ``sys.builtin_module_names`` and have no
    ``__file__`` attribute at all, so reading it would raise
    ``AttributeError``.) What is being pinned is the guard's direction, not a
    provenance: whenever the module cannot name bytes to hash, the identity
    call must raise rather than record ``None``, because a ``None`` there
    would mint a native-lane receipt with no binary bound to it — the exact
    drift the field exists to catch.
    """

    monkeypatch.setattr(simsoptpp, "__file__", None)
    with pytest.raises(RuntimeError, match="cannot hash the extension"):
        nested_ls_runtime_identity()


def _first_type_change(
    original: object, restored: object, path: str
) -> tuple[str, type, type] | None:
    """First path where the round trip changed a value's EXACT type.

    Exact types, not ``isinstance``: ``np.float64`` is a ``float`` subclass
    and ``np.str_`` is a ``str`` subclass, so an ``isinstance`` check would
    wave through precisely the substitutions that silently change what
    consumers read back.
    """

    if type(original) is not type(restored):
        return (path, type(original), type(restored))
    if isinstance(original, dict) and isinstance(restored, dict):
        for key in original:
            found = _first_type_change(original[key], restored[key], f"{path}[{key!r}]")
            if found is not None:
                return found
    elif isinstance(original, list) and isinstance(restored, list):
        for index, item in enumerate(original):
            found = _first_type_change(item, restored[index], f"{path}[{index}]")
            if found is not None:
                return found
    return None


def test_every_identity_value_keeps_its_type_across_dump_strict_json() -> None:
    """No value in the identity block changes type when it is serialized.

    Equality across the round trip is true by construction — ``json.loads`` of
    a ``sort_keys=True`` dump of str-keyed JSON-native values always compares
    equal — so it measures nothing. What can actually go wrong is a value that
    is only JSON-*adjacent*: a tuple that comes back a list, an ``np.float64``
    or ``np.str_`` that comes back a plain ``float``/``str``. Evidence
    consumers read the parsed form, so any such substitution means the
    in-process identity block and the archived one are not the same document.
    """

    identity = nested_ls_runtime_identity()
    restored = json.loads(dump_strict_json(identity))
    changed = _first_type_change(identity, restored, "identity")
    detail = (
        ""
        if changed is None
        else (
            f"{changed[0]} is {changed[1].__name__} in the identity block but "
            f"parses back as {changed[2].__name__}"
        )
    )

    assert changed is None, (
        f"a value does not survive dump_strict_json with its type: {detail}. "
        "Record a JSON-native value there instead, so the in-process identity "
        "block and the archived one are the same document."
    )


def test_receipt_provenance_still_carries_the_native_binding() -> None:
    """The receipt keys must not have been lost to the identity refactor.

    ``nested_ls_receipt_provenance`` used to derive the binding itself; it
    now inherits it from the identity block it already spreads. Archived
    receipts read ``provenance["simsoptpp_sha256"]``, so both keys must still
    be there and must still describe the imported extension.
    """

    provenance = nested_ls_receipt_provenance()
    expected_path = _imported_extension_path()
    assert Path(str(provenance["simsoptpp_path"])) == expected_path
    assert (
        provenance["simsoptpp_sha256"]
        == hashlib.sha256(expected_path.read_bytes()).hexdigest()
    ), (
        "nested_ls_receipt_provenance no longer reports the imported "
        "extension's digest; archived receipts pin this key."
    )
