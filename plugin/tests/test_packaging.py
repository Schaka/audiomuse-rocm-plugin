"""The manifest and the two published catalog files, as core will read them."""

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugin"
MANIFEST = PLUGIN_DIR / "rocm_accelerator" / "plugin.json"
CATALOG_SH = REPO_ROOT / ".github" / "scripts" / "catalog.sh"
PACKAGE_SH = REPO_ROOT / ".github" / "scripts" / "package-plugin.sh"

needs_jq = pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
needs_zip = pytest.mark.skipif(shutil.which("zip") is None, reason="zip not installed")


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


class TestManifest:
    def test_has_the_fields_core_reads(self, manifest):
        for key in ("id", "name", "author", "description", "version",
                    "min_core_version", "targets", "requirements"):
            assert key in manifest, key

    def test_id_matches_the_package_directory(self, manifest):
        # Core imports the plugin by id, so the two cannot drift apart.
        assert manifest["id"] == "rocm_accelerator"
        assert (PLUGIN_DIR / manifest["id"] / "__init__.py").exists()

    def test_declares_no_pip_requirements(self, manifest):
        # A PyPI onnxruntime pulled in as a dependency would replace the image's
        # MIGraphX build with a CPU-only one.
        assert manifest["requirements"] == []

    def test_targets_the_worker(self, manifest):
        # Everything it registers runs on the worker; a flask-side install would
        # find no GPU and nothing to accelerate.
        assert manifest["targets"] == ["worker"]


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    """The two files catalog.sh publishes, parsed."""
    outdir = tmp_path_factory.mktemp("catalog")
    versions = outdir / "versions.json"
    versions.write_text(json.dumps([{
        "version": "1.2.3",
        "min_core_version": "3.1.0",
        "changelog": "test",
        "sourceUrl": "https://example.invalid/rocm_accelerator.zip",
        "checksum": "d41d8cd98f00b204e9800998ecf8427e",
    }]))
    subprocess.run(
        [str(CATALOG_SH), str(versions), "https://example.invalid/plugin.json", str(outdir)],
        cwd=REPO_ROOT, check=True,
    )
    return (
        json.loads((outdir / "repository.json").read_text()),
        json.loads((outdir / "plugin.json").read_text()),
    )


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    """The installable zip's entry names, plus its in-archive manifest."""
    out = tmp_path_factory.mktemp("zip") / "rocm_accelerator.zip"
    subprocess.run([str(PACKAGE_SH), "1.2.3", str(out)], cwd=REPO_ROOT, check=True)
    with zipfile.ZipFile(out) as archive:
        return sorted(archive.namelist()), archive.read("rocm_accelerator/plugin.json")


@needs_jq
class TestCatalog:
    def test_repository_is_a_plugins_list(self, catalog):
        repository, _ = catalog

        assert isinstance(repository.get("plugins"), list)
        assert len(repository["plugins"]) == 1

    def test_repository_entry_points_at_the_plugin_document(self, catalog, manifest):
        repository, _ = catalog
        entry = repository["plugins"][0]

        assert entry["pluginUrl"] == "https://example.invalid/plugin.json"
        assert entry["id"] == manifest["id"]
        assert entry["name"] == manifest["name"]

    def test_plugin_document_carries_the_versions(self, catalog):
        _, plugin_doc = catalog

        assert [v["version"] for v in plugin_doc["versions"]] == ["1.2.3"]

    def test_every_version_has_a_source_url(self, catalog):
        # Core silently drops a version without one, which would make the plugin
        # vanish from the catalog with no error anywhere.
        _, plugin_doc = catalog

        for version in plugin_doc["versions"]:
            assert version.get("sourceUrl")

    def test_targets_and_requirements_sit_at_the_top_level(self, catalog, manifest):
        # Core reads them from the document, not from the version entries.
        _, plugin_doc = catalog

        assert plugin_doc["targets"] == manifest["targets"]
        assert plugin_doc["requirements"] == manifest["requirements"]


@needs_jq
@needs_zip
class TestZip:
    def test_has_one_top_level_directory_holding_the_manifest(self, package):
        names, _ = package

        assert "rocm_accelerator/plugin.json" in names
        assert {name.split("/")[0] for name in names} == {"rocm_accelerator"}

    def test_ships_every_python_module(self, package):
        names, _ = package
        source_dir = PLUGIN_DIR / "rocm_accelerator"
        expected = {
            f"rocm_accelerator/{path.relative_to(source_dir)}"
            for path in source_dir.rglob("*.py")
            if "__pycache__" not in path.parts
        }

        assert expected <= set(names), expected - set(names)

    def test_version_is_stamped_into_the_manifest(self, package):
        _, manifest_bytes = package

        assert json.loads(manifest_bytes)["version"] == "1.2.3"

    def test_no_bytecode_is_shipped(self, package):
        names, _ = package

        assert not [name for name in names if name.endswith(".pyc") or "__pycache__" in name]
