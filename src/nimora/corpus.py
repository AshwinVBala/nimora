from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


@dataclass(slots=True)
class CorpusPolicy:
    allowed_licenses: list[str] = field(
        default_factory=lambda: [
            "Apache-2.0",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "ISC",
            "MIT",
        ]
    )
    allowed_extensions: list[str] = field(
        default_factory=lambda: [
            ".c",
            ".cc",
            ".cpp",
            ".cs",
            ".css",
            ".go",
            ".h",
            ".hpp",
            ".html",
            ".java",
            ".js",
            ".json",
            ".jsx",
            ".kt",
            ".kts",
            ".md",
            ".php",
            ".proto",
            ".py",
            ".rb",
            ".rs",
            ".rst",
            ".scala",
            ".sh",
            ".sql",
            ".swift",
            ".toml",
            ".ts",
            ".tsx",
            ".xml",
            ".yaml",
            ".yml",
        ]
    )
    allowed_filenames: list[str] = field(
        default_factory=lambda: ["Dockerfile", "Makefile"]
    )
    excluded_globs: list[str] = field(
        default_factory=lambda: [
            "**/.env*",
            "**/*.lock",
            "**/*.map",
            "**/*.min.css",
            "**/*.min.js",
            "**/*_generated.*",
            "**/*.generated.*",
            "**/.venv/**",
            "**/__pycache__/**",
            "**/build/**",
            "**/coverage/**",
            "**/dist/**",
            "**/node_modules/**",
            "**/package-lock.json",
            "**/target/**",
            "**/vendor/**",
        ]
    )
    min_file_chars: int = 32
    max_file_bytes: int = 1_000_000
    validation_fraction: float = 0.02
    split_salt: str = "nimora-corpus-v1"
    near_duplicate_hamming_distance: int = 3
    require_clean_git: bool = True
    require_pinned_revision: bool = True
    reject_sensitive_content: bool = True

    def validate(self) -> None:
        if not self.allowed_licenses:
            raise ValueError("allowed_licenses cannot be empty")
        if self.min_file_chars < 1 or self.max_file_bytes < 1:
            raise ValueError("file-size limits must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if not 0 <= self.near_duplicate_hamming_distance <= 3:
            raise ValueError("near_duplicate_hamming_distance must be in [0, 3]")
        if not self.split_salt:
            raise ValueError("split_salt cannot be empty")


@dataclass(frozen=True, slots=True)
class CorpusSource:
    source_id: str
    path: Path
    source_url: str
    license_spdx: str
    license_file: PurePosixPath
    revision: str | None
    authorized: bool


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    source: CorpusSource
    revision: str
    license_sha256: str
    split: str


SENSITIVE_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,255}\b"),
    "github_fine_grained_token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,255}\b"),
    "hugging_face_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{40,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "stripe_secret": re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "email_address": re.compile(
        r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z])"
    ),
}

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]")


def load_policy(path: str | Path) -> CorpusPolicy:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    policy = CorpusPolicy(**values)
    policy.validate()
    return policy


def load_sources(path: str | Path) -> list[CorpusSource]:
    sources: list[CorpusSource] = []
    identifiers: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            required = {
                "authorized",
                "license_file",
                "license_spdx",
                "path",
                "source_id",
                "source_url",
            }
            missing = sorted(required - record.keys())
            if missing:
                raise ValueError(f"Missing {missing} at {path}:{line_number}")
            source_id = str(record["source_id"]).strip()
            if not source_id or source_id in identifiers:
                raise ValueError(f"Invalid or duplicate source_id at {path}:{line_number}")
            source_url = str(record["source_url"]).strip()
            if not source_url:
                raise ValueError(f"source_url is empty at {path}:{line_number}")
            license_file = PurePosixPath(str(record["license_file"]))
            if license_file.is_absolute() or ".." in license_file.parts:
                raise ValueError(f"license_file is unsafe at {path}:{line_number}")
            revision = str(record["revision"]) if record.get("revision") else None
            if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError(f"revision must be a full Git SHA at {path}:{line_number}")
            identifiers.add(source_id)
            sources.append(
                CorpusSource(
                    source_id=source_id,
                    path=Path(record["path"]).expanduser().resolve(),
                    source_url=source_url,
                    license_spdx=str(record["license_spdx"]),
                    license_file=license_file,
                    revision=revision,
                    authorized=record["authorized"] is True,
                )
            )
    if not sources:
        raise ValueError("The source manifest is empty")
    return sorted(sources, key=lambda item: item.source_id)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"Git command failed for {repository}: {detail}")
    return completed.stdout


def _source_split_score(source_id: str, policy: CorpusPolicy) -> bytes:
    return hashlib.sha256(f"{policy.split_salt}\0{source_id}".encode()).digest()


def resolve_source(source: CorpusSource, policy: CorpusPolicy) -> ResolvedSource:
    if not source.authorized:
        raise ValueError(f"Source {source.source_id!r} is not explicitly authorized")
    if source.license_spdx not in policy.allowed_licenses:
        raise ValueError(
            f"Source {source.source_id!r} uses disallowed license {source.license_spdx!r}"
        )
    if not source.path.is_dir():
        raise ValueError(f"Source path is not a directory: {source.path}")
    revision = _git(source.path, "rev-parse", "HEAD").strip()
    if policy.require_pinned_revision and source.revision is None:
        raise ValueError(f"Source {source.source_id!r} must pin a revision")
    if source.revision is not None and revision != source.revision:
        raise ValueError(
            f"Source {source.source_id!r} is at {revision}, expected {source.revision}"
        )
    if policy.require_clean_git and _git(source.path, "status", "--porcelain").strip():
        raise ValueError(f"Source {source.source_id!r} has uncommitted changes")
    tracked = set(tracked_files(source.path))
    if source.license_file not in tracked:
        raise ValueError(
            f"License evidence {source.license_file} is not tracked in {source.source_id!r}"
        )
    license_path = source.path.joinpath(*source.license_file.parts)
    if license_path.is_symlink() or not license_path.is_file():
        raise ValueError(f"License evidence is not a regular file: {license_path}")
    license_sha256 = _file_sha256(license_path)
    return ResolvedSource(source, revision, license_sha256, "")


def assign_repository_splits(
    sources: list[ResolvedSource], policy: CorpusPolicy
) -> list[ResolvedSource]:
    if len(sources) < 2:
        raise ValueError("At least two source repositories are required for a safe split")
    validation_count = max(1, round(len(sources) * policy.validation_fraction))
    validation_count = min(validation_count, len(sources) - 1)
    ranked = sorted(
        sources,
        key=lambda item: (
            _source_split_score(item.source.source_id, policy),
            item.source.source_id,
        ),
    )
    validation_ids = {
        item.source.source_id for item in ranked[:validation_count]
    }
    return [
        ResolvedSource(
            source=item.source,
            revision=item.revision,
            license_sha256=item.license_sha256,
            split=(
                "validation" if item.source.source_id in validation_ids else "train"
            ),
        )
        for item in sources
    ]


def tracked_files(repository: Path) -> list[PurePosixPath]:
    raw = subprocess.run(
        ["git", "-C", str(repository), "ls-files", "-z"],
        check=False,
        capture_output=True,
        timeout=300,
    )
    if raw.returncode:
        raise ValueError(f"Could not list tracked files in {repository}")
    paths = sorted(
        PurePosixPath(value.decode("utf-8"))
        for value in raw.stdout.split(b"\0")
        if value
    )
    if any(path.is_absolute() or ".." in path.parts for path in paths):
        raise ValueError(f"Repository contains an unsafe tracked path: {repository}")
    return paths


def _matches_glob(path: str, pattern: str) -> bool:
    if fnmatch.fnmatch(path, pattern):
        return True
    return pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:])


def file_allowed(path: PurePosixPath, policy: CorpusPolicy) -> bool:
    path_text = path.as_posix()
    if any(_matches_glob(path_text, pattern) for pattern in policy.excluded_globs):
        return False
    return path.name in policy.allowed_filenames or path.suffix.lower() in {
        suffix.lower() for suffix in policy.allowed_extensions
    }


def normalize_text(raw: bytes, policy: CorpusPolicy) -> tuple[str | None, str | None]:
    if len(raw) > policy.max_file_bytes:
        return None, "file_too_large"
    if b"\0" in raw:
        return None, "binary_file"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, "invalid_utf8"
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) < policy.min_file_chars:
        return None, "file_too_short"
    if policy.reject_sensitive_content:
        for name, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                return None, f"sensitive:{name}"
    return text + "\n", None


def read_candidate(path: Path, maximum_bytes: int) -> tuple[bytes | None, str | None]:
    with path.open("rb") as handle:
        raw = handle.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        return None, "file_too_large"
    return raw, None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def simhash64(text: str, maximum_features: int = 20_000) -> int:
    tokens = TOKEN_PATTERN.findall(text)
    if not tokens:
        return 0
    normalized = ["<NUM>" if token.isdigit() else token for token in tokens]
    stride = max(1, len(normalized) // maximum_features)
    vector = [0] * 64
    for feature_number, index in enumerate(range(0, len(normalized), stride)):
        if feature_number >= maximum_features:
            break
        feature = "\0".join(normalized[index : index + 5])
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return result


class DuplicateIndex:
    def __init__(self, hamming_distance: int) -> None:
        self.hamming_distance = hamming_distance
        self.exact: dict[str, str] = {}
        self.hashes: list[int] = []
        self.identities: list[str] = []
        self.buckets: dict[tuple[int, int], list[int]] = {}

    def add(self, identity: str, sha256: str, simhash: int) -> tuple[bool, str | None]:
        if sha256 in self.exact:
            return False, self.exact[sha256]
        candidates: set[int] = set()
        for band in range(4):
            bucket = (band, (simhash >> (band * 16)) & 0xFFFF)
            candidates.update(self.buckets.get(bucket, []))
        for candidate in sorted(candidates):
            if (simhash ^ self.hashes[candidate]).bit_count() <= self.hamming_distance:
                return False, self.identities[candidate]
        index = len(self.hashes)
        self.exact[sha256] = identity
        self.hashes.append(simhash)
        self.identities.append(identity)
        for band in range(4):
            bucket = (band, (simhash >> (band * 16)) & 0xFFFF)
            self.buckets.setdefault(bucket, []).append(index)
        return True, None


def _write_jsonl(handle, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _policy_digest(policy: CorpusPolicy) -> str:
    encoded = json.dumps(asdict(policy), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_source_stable(source: ResolvedSource, policy: CorpusPolicy) -> None:
    current_revision = _git(source.source.path, "rev-parse", "HEAD").strip()
    if current_revision != source.revision:
        raise ValueError(f"Source changed revision during build: {source.source.source_id}")
    if policy.require_clean_git and _git(
        source.source.path, "status", "--porcelain"
    ).strip():
        raise ValueError(f"Source changed during build: {source.source.source_id}")


def build_corpus(
    sources_path: str | Path,
    policy_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    policy = load_policy(policy_path)
    resolved_sources = [
        resolve_source(source, policy) for source in load_sources(sources_path)
    ]
    sources = assign_repository_splits(resolved_sources, policy)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    duplicate_index = DuplicateIndex(policy.near_duplicate_hamming_distance)
    counts: dict[str, int] = {
        "accepted": 0,
        "rejected": 0,
        "train": 0,
        "validation": 0,
    }
    rejection_reasons: dict[str, int] = {}

    try:
        with (
            (staging / "train.jsonl").open("w", encoding="utf-8") as train_handle,
            (staging / "validation.jsonl").open("w", encoding="utf-8") as validation_handle,
            (staging / "audit.jsonl").open("w", encoding="utf-8") as audit_handle,
        ):
            output_handles = {"train": train_handle, "validation": validation_handle}
            for resolved in sources:
                source = resolved.source
                for relative in tracked_files(source.path):
                    identity = f"{source.source_id}:{relative.as_posix()}"
                    audit: dict[str, Any] = {
                        "source_id": source.source_id,
                        "path": relative.as_posix(),
                    }
                    reason: str | None = None
                    if not file_allowed(relative, policy):
                        reason = "path_filtered"
                    else:
                        absolute = source.path.joinpath(*relative.parts)
                        if absolute.is_symlink() or not absolute.is_file():
                            reason = "not_regular_file"
                        else:
                            raw, reason = read_candidate(
                                absolute, policy.max_file_bytes
                            )
                            text = None
                            if raw is not None:
                                text, reason = normalize_text(raw, policy)
                            if text is not None:
                                sha256 = content_sha256(text)
                                accepted, duplicate_of = duplicate_index.add(
                                    identity, sha256, simhash64(text)
                                )
                                if not accepted:
                                    reason = "duplicate"
                                    audit["duplicate_of"] = duplicate_of
                                else:
                                    record = {
                                        "text": text,
                                        "metadata": {
                                            "license_spdx": source.license_spdx,
                                            "license_file": source.license_file.as_posix(),
                                            "license_sha256": resolved.license_sha256,
                                            "path": relative.as_posix(),
                                            "revision": resolved.revision,
                                            "sha256": sha256,
                                            "source_id": source.source_id,
                                            "source_url": source.source_url,
                                            "split": resolved.split,
                                        },
                                    }
                                    _write_jsonl(output_handles[resolved.split], record)
                                    audit.update(
                                        decision="accepted",
                                        sha256=sha256,
                                        split=resolved.split,
                                    )
                                    counts["accepted"] += 1
                                    counts[resolved.split] += 1
                    if reason is not None:
                        audit.update(decision="rejected", reason=reason)
                        counts["rejected"] += 1
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    _write_jsonl(audit_handle, audit)
                _assert_source_stable(resolved, policy)

        if counts["train"] == 0 or counts["validation"] == 0:
            raise ValueError(
                "Filtering produced an empty train or validation split; add eligible sources"
            )

        source_lock = [
            {
                "authorized": item.source.authorized,
                "license_spdx": item.source.license_spdx,
                "license_file": item.source.license_file.as_posix(),
                "license_sha256": item.license_sha256,
                "path": str(item.source.path),
                "revision": item.revision,
                "source_id": item.source.source_id,
                "source_url": item.source.source_url,
                "split": item.split,
            }
            for item in sources
        ]
        manifest = {
            "format_version": 1,
            "policy_sha256": _policy_digest(policy),
            "counts": counts,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "sources": source_lock,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
