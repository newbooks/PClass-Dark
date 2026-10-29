"""Migrate a selected library to clusters/<cluster_id>/ with an original backup."""

import argparse
import json
from pathlib import Path

from pclass_dark.library_paths import migrate_library_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate_library_paths(args.library_root), indent=2))
