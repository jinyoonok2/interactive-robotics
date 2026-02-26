"""
Object Catalog for Affordance Pipeline
=======================================
Loads object definitions from config/objects.json.

Each object has:
  - ycb_config: path to YCB object config (relative to habitat-lab/)
  - display_name: human-readable name
  - parts: dict of part_name → description
  - offset: [x, y, z] spawn offset from agent position

To add a new object, edit:
  affordance-pipeline/config/objects.json
"""

import json
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# LOAD OBJECT CATALOG FROM JSON
# ═══════════════════════════════════════════════════════════════════════════

_CONFIG_DIR = Path(__file__).resolve().parent / "config"
_OBJECTS_PATH = _CONFIG_DIR / "objects.json"


def _load_objects() -> dict:
    """Load and cache the object catalog from objects.json."""
    if not _OBJECTS_PATH.exists():
        raise FileNotFoundError(
            f"Object catalog not found at {_OBJECTS_PATH}\n"
            f"Create it or copy from the template."
        )
    with open(_OBJECTS_PATH, "r") as f:
        data = json.load(f)
    # Strip JSON comment keys
    return {k: v for k, v in data.items() if not k.startswith("_")}


# Load once at import time
OBJECTS = _load_objects()


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_object_names():
    """Return list of available object names."""
    return list(OBJECTS.keys())


def get_object(name):
    """Get object config by name. Raises ValueError if not found."""
    if name not in OBJECTS:
        available = ", ".join(OBJECTS.keys())
        raise ValueError(f"Unknown object '{name}'. Available: {available}")
    return OBJECTS[name]


def get_parts(name):
    """Return list of available part names for an object."""
    obj = get_object(name)
    return list(obj["parts"].keys())


def validate_part(obj_name, part_name):
    """Validate that a part exists for the given object. Raises ValueError if not."""
    obj = get_object(obj_name)
    if part_name not in obj["parts"]:
        available = ", ".join(obj["parts"].keys())
        raise ValueError(
            f"Unknown part '{part_name}' for {obj_name}. Available: {available}"
        )
    return True


def print_catalog():
    """Print the full object catalog."""
    print("\n  Available Objects:")
    print("  " + "─" * 56)
    for name, obj in OBJECTS.items():
        parts_str = ", ".join(obj["parts"].keys())
        print(f"    {name:14s} │ {obj['display_name']:25s} │ {parts_str}")
    print("  " + "─" * 56)


def print_object_parts(obj_name):
    """Print parts for a specific object."""
    obj = get_object(obj_name)
    print(f"\n  Parts for {obj['display_name']}:")
    print("  " + "─" * 44)
    for part, desc in obj["parts"].items():
        print(f"    {part:12s} │ {desc}")
    print("  " + "─" * 44)
