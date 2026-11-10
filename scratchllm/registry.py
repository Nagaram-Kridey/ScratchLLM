"""Dataset registry: where each dataset came from, its licence, and its file hashes."""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path


class RegistryError(Exception):
    pass


@dataclass(frozen=True)
class DataFile:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Dataset:
    id: str
    url: str
    revision: str
    licence: str
    licence_url: str
    files: tuple[DataFile, ...]


_REQUIRED = ("id", "url", "revision", "licence", "licence_url")


def load_registry(path) -> list[Dataset]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    datasets, seen = [], set()
    for entry in raw.get("dataset", []):
        ident = entry.get("id") or "<no id>"
        for field in _REQUIRED:
            if not str(entry.get(field, "")).strip():
                raise RegistryError(f"dataset {ident}: {field} is missing or empty")
        if ident in seen:
            raise RegistryError(f"dataset {ident}: id appears more than once")
        seen.add(ident)
        try:
            files = tuple(
                DataFile(f["name"], int(f["size"]), f["sha256"].lower()) for f in entry.get("file", [])
            )
        except KeyError as missing:
            raise RegistryError(f"dataset {ident}: a file entry is missing {missing}") from None
        datasets.append(Dataset(*(entry[f] for f in _REQUIRED), files))
    return datasets


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify(ds: Dataset, raw_dir) -> None:
    """Check every file of a dataset at raw_dir/<id>/<name>. Raises RegistryError."""
    for file in ds.files:
        path = Path(raw_dir) / ds.id / file.name
        if not path.is_file():
            raise RegistryError(f"{ds.id}/{file.name}: missing at {path}")
        size = path.stat().st_size
        if size != file.size:
            raise RegistryError(f"{ds.id}/{file.name}: size is {size}, registry says {file.size}")
        actual = sha256_file(path)
        if actual != file.sha256:
            raise RegistryError(f"{ds.id}/{file.name}: sha256 is {actual}, registry says {file.sha256}")


def render_licences(datasets: list[Dataset]) -> str:
    lines = [
        "# Dataset licences",
        "",
        "Generated from `data/registry.toml` by `python -m scratchllm licences`. Do not edit by hand.",
        "",
        "| Dataset | Licence | Licence text | Source | Revision |",
        "|---|---|---|---|---|",
    ]
    for ds in sorted(datasets, key=lambda d: d.id):
        lines.append(f"| {ds.id} | {ds.licence} | {ds.licence_url} | {ds.url} | {ds.revision} |")
    return "\n".join(lines) + "\n"
