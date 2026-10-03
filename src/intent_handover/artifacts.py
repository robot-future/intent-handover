"""Publish JSON atomically and invalidate completion indexes during reruns."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile


def write_json(path, data):
    path = Path(path)
    payload = json.dumps(data, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            name = stream.name
            stream.write(payload)
        os.replace(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


@contextmanager
def export_run(output, indexes):
    """Old files stay available, but incomplete indexes cannot pass v1 readers.

    Callers publish their real schema only after all referenced files exist.
    One output directory is intended for one writer at a time.
    """
    output = Path(output)
    marker = {"schema_version": "handover.incomplete.v1", "status": "running"}
    for name in indexes:
        write_json(output/name, marker)
    try:
        yield
    except BaseException as exc:
        marker.update(status="failed", error=str(exc))
        for name in indexes:
            write_json(output/name, marker)
        raise
