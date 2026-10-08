"""Expose an existing output directory in the repository's unified results tree."""
import argparse
from pathlib import Path


def register(section: str, target: Path, name: str | None = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    target = target.resolve(strict=True)
    if not target.is_dir():
        raise ValueError(f'Output must be a directory: {target}')
    name = name or target.name
    if not name or name in {'.', '..'} or Path(name).name != name:
        raise ValueError('Result name must be a single directory name')
    link = root / 'results' / section / name
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == target:
        return link
    if link.exists() or link.is_symlink():
        raise FileExistsError(f'Refusing to replace an existing result: {link}')
    link.symlink_to(target, target_is_directory=True)
    return link


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section', required=True, choices=[
        'part2action/runs', 'part2action/evaluations', 'part2action/archive',
        'partinstruct/runs', 'partinstruct/evaluations', 'partinstruct/archive', 'diagnostics', 'data'])
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--name')
    args = parser.parse_args()
    print(register(args.section, args.target, args.name))
