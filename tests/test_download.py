"""The download script's selection rule and manifest, with Hugging Face replaced by a stand-in."""

import json
import runpy
import sys
import types
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_demo_data.py"


def fake_hub(tmp_path):
    files = {f"data/mdcath_dataset_{name}.h5": size for name, size in [("3big000", 900), ("1aaaA00", 300), ("2bbbA00", 100), ("4cccA00", 200)]}

    class Entry:
        def __init__(self, path, size):
            self.path, self.size = path, size

    class HfApi:
        def dataset_info(self, repo):
            return types.SimpleNamespace(sha="abc123")

        def list_repo_tree(self, repo, path_in_repo, repo_type, revision):
            return [Entry(p, s) for p, s in files.items()] + [Entry("data/README.txt", 1)]

    def hf_hub_download(repo, filename, repo_type, revision, local_dir):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * files[filename])
        return str(target)

    return types.SimpleNamespace(HfApi=HfApi, hf_hub_download=hf_hub_download)


def test_smallest_files_rule_and_manifest(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub(tmp_path))
    monkeypatch.setattr(sys, "argv", ["download_demo_data.py", "--n", "2", "--root", str(tmp_path)])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert [Path(f["path"]).name for f in manifest["files"]] == ["mdcath_dataset_2bbbA00.h5", "mdcath_dataset_4cccA00.h5"]
    assert manifest["revision"] == "abc123" and "2 smallest" in manifest["selection_rule"]
    assert all(len(f["sha256"]) == 64 for f in manifest["files"])
