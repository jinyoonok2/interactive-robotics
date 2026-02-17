"""
Object Catalog for Affordance Pipeline
=======================================
Defines available objects, their YCB configs, and graspable parts.

Each object has:
  - ycb_config: path to YCB object config (relative to habitat-lab/)
  - display_name: human-readable name
  - parts: dict of part_name → description
  - offset: [x, y, z] spawn offset from agent position
"""

# ═══════════════════════════════════════════════════════════════════════════
# OBJECT CATALOG
# ═══════════════════════════════════════════════════════════════════════════

OBJECTS = {
    "mug": {
        "ycb_config": "data/objects/ycb/configs/025_mug.object_config.json",
        "display_name": "Coffee Mug (YCB 025)",
        "parts": {
            "handle": "Protruding loop on the side for gripping",
            "body":   "Main cylindrical cup body",
            "rim":    "Circular top edge of the cup",
        },
        "offset": [0.0, 0.6, -1.0],
    },
    "power_drill": {
        "ycb_config": "data/objects/ycb/configs/035_power_drill.object_config.json",
        "display_name": "Power Drill (YCB 035)",
        "parts": {
            "handle": "Pistol grip for holding",
            "body":   "Main motor housing",
            "chuck":  "Front end where the drill bit attaches",
        },
        "offset": [0.0, 0.6, -1.0],
    },
    "pitcher": {
        "ycb_config": "data/objects/ycb/configs/019_pitcher_base.object_config.json",
        "display_name": "Pitcher (YCB 019)",
        "parts": {
            "handle": "Large handle on the back",
            "spout":  "Pouring lip at the front",
            "body":   "Main container body",
            "rim":    "Top circular edge",
        },
        "offset": [0.0, 0.6, -1.0],
    },
    "hammer": {
        "ycb_config": "data/objects/ycb/configs/048_hammer.object_config.json",
        "display_name": "Hammer (YCB 048)",
        "parts": {
            "head":   "Metal striking head",
            "handle": "Wooden grip handle",
        },
        "offset": [0.0, 0.6, -1.0],
    },
}


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
