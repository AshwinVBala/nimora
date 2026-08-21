from pathlib import Path, PurePosixPath

import pytest

from nimora.corpus import (
    CorpusPolicy,
    CorpusSource,
    DuplicateIndex,
    ResolvedSource,
    assign_repository_splits,
    content_sha256,
    file_allowed,
    normalize_text,
    read_candidate,
    simhash64,
)


def test_root_and_nested_vendor_paths_are_filtered():
    policy = CorpusPolicy()
    assert not file_allowed(PurePosixPath("vendor/package.py"), policy)
    assert not file_allowed(PurePosixPath("src/vendor/package.py"), policy)
    assert file_allowed(PurePosixPath("src/package.py"), policy)


def test_sensitive_content_is_rejected_without_copying_secret():
    text, reason = normalize_text(
        b"key = 'AKIAABCDEFGHIJKLMNOP'\n",
        CorpusPolicy(min_file_chars=1),
    )
    assert text is None
    assert reason == "sensitive:aws_access_key"


def test_candidate_reads_are_bounded(tmp_path):
    path = tmp_path / "large.py"
    path.write_bytes(b"x" * 32)
    raw, reason = read_candidate(path, maximum_bytes=16)
    assert raw is None
    assert reason == "file_too_large"


def test_exact_duplicate_points_to_canonical_identity():
    text = "def answer():\n    return 42\n"
    digest = content_sha256(text)
    fingerprint = simhash64(text)
    index = DuplicateIndex(hamming_distance=3)
    assert index.add("one:a.py", digest, fingerprint) == (True, None)
    assert index.add("two:b.py", digest, fingerprint) == (False, "one:a.py")


def test_repository_split_is_deterministic_and_nonempty():
    policy = CorpusPolicy(validation_fraction=0.02)
    resolved = []
    for source_id in ("a/project", "b/project", "c/project"):
        source = CorpusSource(
            source_id=source_id,
            path=Path("/tmp") / source_id,
            source_url=f"https://example.test/{source_id}",
            license_spdx="MIT",
            license_file=PurePosixPath("LICENSE"),
            revision="a" * 40,
            authorized=True,
        )
        resolved.append(ResolvedSource(source, "a" * 40, "b" * 64, ""))
    first = assign_repository_splits(resolved, policy)
    second = assign_repository_splits(list(reversed(resolved)), policy)
    first_map = {item.source.source_id: item.split for item in first}
    second_map = {item.source.source_id: item.split for item in second}
    assert first_map == second_map
    assert set(first_map.values()) == {"train", "validation"}


def test_single_repository_cannot_create_leakage_safe_split():
    source = CorpusSource(
        source_id="only/project",
        path=Path("/tmp/only/project"),
        source_url="https://example.test/only/project",
        license_spdx="MIT",
        license_file=PurePosixPath("LICENSE"),
        revision="a" * 40,
        authorized=True,
    )
    resolved = [ResolvedSource(source, "a" * 40, "b" * 64, "")]
    with pytest.raises(ValueError, match="At least two"):
        assign_repository_splits(resolved, CorpusPolicy())
