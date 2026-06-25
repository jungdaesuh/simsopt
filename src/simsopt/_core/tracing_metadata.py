from __future__ import annotations

from threading import RLock
from typing import Optional
from weakref import WeakKeyDictionary, WeakValueDictionary


_LOCK = RLock()
_LEVELSET_CLASSIFIERS = WeakKeyDictionary()
_LEVELSET_INTERPOLANT_CLASSIFIERS = WeakValueDictionary()


def register_levelset_classifier(criterion: object, classifier: object) -> None:
    with _LOCK:
        _LEVELSET_CLASSIFIERS[criterion] = classifier


def levelset_classifier_for(criterion: object) -> Optional[object]:
    with _LOCK:
        return _LEVELSET_CLASSIFIERS.get(criterion)


def register_levelset_interpolant(interpolant: object, classifier: object) -> None:
    with _LOCK:
        _LEVELSET_INTERPOLANT_CLASSIFIERS[id(interpolant)] = classifier


def levelset_classifier_from_interpolant(interpolant: object) -> Optional[object]:
    with _LOCK:
        return _LEVELSET_INTERPOLANT_CLASSIFIERS.get(id(interpolant))
