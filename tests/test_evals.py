import json

import pytest

from nimora.agent.evals import load_eval_cases


def test_eval_manifest_requires_pinned_revision(tmp_path):
    manifest = tmp_path / "cases.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "case_id": "parser",
                "repository": str(tmp_path),
                "revision": "short",
                "task": "Fix parser",
                "checks": [{"argv": ["pytest", "-q"]}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="full Git SHA"):
        load_eval_cases(manifest)
