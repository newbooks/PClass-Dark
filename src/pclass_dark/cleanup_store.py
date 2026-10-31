"""Temporary disk storage for exact inventories and hard-link accounting."""

import json
import sqlite3
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .artifacts import canonical


class InventoryStore(Mapping):
    """Own temporary storage; callers close it after consuming a plan."""

    def __init__(self):
        self.db = None
        self.directory = tempfile.TemporaryDirectory(prefix="pclass-clean-")
        self.db = sqlite3.connect(Path(self.directory.name) / "inventory.sqlite")
        self.db.execute("PRAGMA cache_size=-4096")
        self.db.execute("CREATE TABLE trees (key TEXT PRIMARY KEY, value BLOB)")
        self.db.execute(
            "CREATE TABLE links (device INTEGER, inode INTEGER, count INTEGER, "
            "nlink INTEGER, blocks INTEGER, seen INTEGER DEFAULT 0, "
            "PRIMARY KEY(device,inode))"
        )

    def __getitem__(self, key):
        row = self.db.execute("SELECT value FROM trees WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def __setitem__(self, key, tree):
        self.db.execute("INSERT INTO trees VALUES (?,?)", (key, canonical(tree)))
        self.db.executemany(
            "INSERT INTO links(device,inode,count,nlink,blocks) VALUES (?,?,1,?,?) "
            "ON CONFLICT(device,inode) DO UPDATE SET count=count+1",
            (
                (f["device"], f["inode"], f["nlink"], f["blocks"])
                for f in tree["files"]
                if f["nlink"] > 1
            ),
        )

    def __iter__(self):
        for row in self.db.execute("SELECT key FROM trees ORDER BY key"):
            yield row[0]

    def __len__(self):
        return self.db.execute("SELECT count(*) FROM trees").fetchone()[0]

    def allocated(self, tree):
        total = 0
        unknown = False
        for f in tree["files"]:
            if f["nlink"] == 1:
                blocks = f["blocks"]
            else:
                group = f["device"], f["inode"]
                count, nlink, blocks, seen = self.db.execute(
                    "SELECT count,nlink,blocks,seen FROM links "
                    "WHERE device=? AND inode=?",
                    group,
                ).fetchone()
                if seen:
                    continue
                self.db.execute(
                    "UPDATE links SET seen=1 WHERE device=? AND inode=?", group
                )
                if count < nlink:
                    continue
            if blocks is None:
                unknown = True
            else:
                total += blocks * 512
        return None if unknown else total

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
            self.directory.cleanup()

    def __del__(self):
        self.close()
