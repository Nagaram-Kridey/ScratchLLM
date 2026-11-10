import hashlib

import pytest

from scratchllm.registry import RegistryError, load_registry, render_licences, sha256_file, verify

CONTENT = b"once upon a time\n"
ENTRY = """
[[dataset]]
id = "{id}"
url = "https://example.org/stories"
revision = "{revision}"
licence = "{licence}"
licence_url = "{licence_url}"

[[dataset.file]]
name = "stories.txt"
size = {size}
sha256 = "{sha}"
"""


def _registry(tmp_path, *entries):
    path = tmp_path / "registry.toml"
    path.write_text("".join(entries))
    return path


def _entry(**over):
    values = dict(
        id="stories", revision="abc123", licence="MIT", licence_url="https://example.org/licence",
        size=len(CONTENT), sha=hashlib.sha256(CONTENT).hexdigest(),
    )
    values.update(over)
    return ENTRY.format(**values)


def _raw(tmp_path, content=CONTENT):
    folder = tmp_path / "raw" / "stories"
    folder.mkdir(parents=True)
    (folder / "stories.txt").write_bytes(content)
    return tmp_path / "raw"


def test_valid_registry_loads(tmp_path):
    (ds,) = load_registry(_registry(tmp_path, _entry()))
    assert ds.id == "stories" and ds.licence == "MIT"
    assert ds.files[0].name == "stories.txt" and ds.files[0].size == len(CONTENT)


def test_empty_registry_loads(tmp_path):
    assert load_registry(_registry(tmp_path, "")) == []


@pytest.mark.parametrize("field", ["licence", "licence_url", "revision"])
def test_empty_field_rejected(tmp_path, field):
    with pytest.raises(RegistryError, match=f"stories.*{field}"):
        load_registry(_registry(tmp_path, _entry(**{field: ""})))


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(RegistryError, match="stories"):
        load_registry(_registry(tmp_path, _entry(), _entry()))


def test_verify_passes(tmp_path):
    (ds,) = load_registry(_registry(tmp_path, _entry()))
    verify(ds, _raw(tmp_path))
    assert sha256_file(tmp_path / "raw" / "stories" / "stories.txt") == ds.files[0].sha256


def test_verify_wrong_size(tmp_path):
    (ds,) = load_registry(_registry(tmp_path, _entry(size=3)))
    with pytest.raises(RegistryError, match="stories.txt.*size"):
        verify(ds, _raw(tmp_path))


def test_verify_wrong_hash(tmp_path):
    (ds,) = load_registry(_registry(tmp_path, _entry()))
    with pytest.raises(RegistryError, match="stories.txt.*sha256"):
        verify(ds, _raw(tmp_path, b"once upon a tiMe\n"))


def test_verify_missing_file(tmp_path):
    (ds,) = load_registry(_registry(tmp_path, _entry()))
    with pytest.raises(RegistryError, match="stories.txt.*missing"):
        verify(ds, tmp_path / "raw")


def test_render_licences(tmp_path):
    datasets = load_registry(_registry(tmp_path, _entry(), _entry(id="wiki", licence="CC0-1.0")))
    table = render_licences(datasets)
    for word in ("stories", "wiki", "MIT", "CC0-1.0", "https://example.org/licence", "abc123"):
        assert word in table
    assert table == render_licences(datasets)
