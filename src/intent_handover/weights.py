"""Download and verify the exact upstream weights used by the release."""
import hashlib
from importlib.resources import files
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()


def catalog():
    return json.loads(files(__package__).joinpath("assets", "weights.json").read_text())


def verify_weights(directory):
    result = {}
    for name, record in catalog().items():
        path = Path(directory)/f"{name}.pth"
        if not path.is_file(): raise ValueError(f"Missing {path}; run intent-handover download-weights")
        actual = sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"Checksum mismatch for {path}; re-download with download-weights --force")
        result[name] = {**record, "bytes": path.stat().st_size}
    return result


def download_weights(output, force=False):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for name, record in catalog().items():
        path = output/f"{name}.pth"
        if path.is_file() and not force:
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Checksum mismatch for {path}; use --force to replace")
            continue
        import gdown
        temporary = output/f"{name}.download"
        try:
            downloaded = gdown.download(id=record["google_drive_id"], output=str(temporary), quiet=False)
            if downloaded is None or not temporary.is_file() or sha256(temporary) != record["sha256"]:
                raise ValueError(f"Download/checksum failed: {name}; original checkpoint left unchanged")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    manifest = verify_weights(output)
    (output/"downloads.json").write_text(json.dumps(manifest, indent=2))
    return manifest
