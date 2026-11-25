"""Check preserved model/config/entrypoint bytes without importing any model.

Run from the repository root:
    PYTHONPATH=part2action python3 -m models.part_grounded_diffusion.verify_sources
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path(__file__).with_name('preserved_sources.json')


def verify_legacy_sources(*, root=ROOT, manifest_path=MANIFEST, only=None):
    manifest = json.loads(Path(manifest_path).read_text())
    sources = manifest['source_sha256']
    if only is not None:
        unknown = set(only) - sources.keys()
        if unknown:
            raise ValueError('Unrecorded legacy source: '+', '.join(sorted(unknown)))
        sources = {key: sources[key] for key in only}
    missing, changed = [], []
    for name, expected in sources.items():
        path = Path(root) / name
        if not path.is_file():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(name)
    if missing or changed:
        raise RuntimeError('Preserved legacy sources differ: '+json.dumps({'missing': missing, 'changed': changed}))
    return {'status': 'passed', 'preserved_sources': len(sources),
            'scope': manifest['scope']}


if __name__ == '__main__':
    print(json.dumps(verify_legacy_sources(), indent=2))
