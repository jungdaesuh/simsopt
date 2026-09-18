"""Where the nested-LS benchmark family writes its run receipts.

``72e7a72b0`` deleted ``docs/receipts/`` from the tree, so every default
naming it pointed at a directory a default run can no longer write into.
One constant owns the replacement, and the clean-tree exemption in
``nested_ls_shamanskii_attribution.git_implementation_dirty`` is derived
from the same constant, so a destination move cannot leave the exemption
behind and make a driver refuse the tree it just wrote its own evidence
into.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Root every live benchmark already writes run output under. It also holds
#: caches and scratch of every other kind, so it is a location, not a licence:
#: the clean-tree gates exempt untracked receipts at ``EVIDENCE`` alone, and
#: everything else under here -- tracked or not -- still dirties the tree.
ARTIFACT_ROOT = REPO / ".artifacts"

#: One receipt directory for the whole nested-LS family, not one per driver:
#: these drivers read each other's receipts by file name (the
#: ``nested_ls_outer_*`` and ``nested_ls_reduced_*`` stems), which is the
#: flat namespace ``docs/receipts/evidence/`` used to provide.
EVIDENCE = ARTIFACT_ROOT / "nested-ls-evidence"
