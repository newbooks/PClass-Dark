"""Read-only, schema-aware cleanup planning; all callers own both root guards."""

import json
import logging
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .artifacts import (
    IntegrityError,
    canonical,
    identity,
    safe_file,
    sha256,
    verify_files,
)
from .cleanup_io import UnsafeTree, inventory, read_metadata
from .cleanup_store import InventoryStore
from .cleanup_types import CleanupEntry, CleanupPlan

LOG = logging.getLogger(__name__)

HEX = re.compile(r"[0-9a-f]{64}\Z")


@dataclass
class Node:
    root: str
    path: str
    kind: str
    metadata: dict
    dependencies: set = field(default_factory=set)
    owners: set = field(default_factory=set)
    protected: str | None = None
    spool: object = field(default=None, repr=False)
    tree_offset: tuple | None = None

    @property
    def tree(self):
        if self.tree_offset is None:
            return None
        offset, length = self.tree_offset
        self.spool.seek(offset)
        return json.loads(self.spool.read(length))

    @tree.setter
    def tree(self, value):
        if value is None:
            self.tree_offset = None
            return
        data = canonical(value)
        self.spool.seek(0, 2)
        self.tree_offset = (self.spool.tell(), len(data))
        self.spool.write(data)

    @property
    def key(self):
        return self.root + ":" + self.path


class Planner:
    def __init__(self, data, library):
        self.roots = {"setup": data, "library": library}
        self.nodes, self.snapshots = {}, {}
        self.prepared, self.embeddings, self.production = {}, {}, {}
        self.companions, self.research, self.fit_owners = {}, {}, defaultdict(set)
        self.fit_nodes, self.target_nodes = {}, []
        self.fold_maps = {}
        self.spool = tempfile.TemporaryFile(mode="w+b")

    def metadata(self, root, relative):
        path = safe_file(self.roots[root], relative)
        value = read_metadata(path)
        self.snapshots[root, relative] = sha256(path)
        return value

    def add(self, root, relative, kind, metadata, expected=None):
        node = Node(root, relative, kind, metadata, spool=self.spool)
        path = safe_file(self.roots[root], relative)
        try:
            tree = inventory(path)
            if tree["directories"][0]["device"] != self.roots[root].stat().st_dev:
                raise UnsafeTree("Candidate is on another device")
            names = {f["path"] for f in tree["files"]}
            expected_dirs = {"."}
            if expected is not None:
                for name in expected:
                    parent = Path(name).parent
                    while parent.as_posix() != ".":
                        expected_dirs.add(parent.as_posix())
                        parent = parent.parent
                if (
                    names != set(expected)
                    or {d["path"] for d in tree["directories"]} != expected_dirs
                ):
                    node.protected = "unexpected files or directories"
            node.tree = tree
        except UnsafeTree as exc:
            node.protected, node.tree = str(exc), None
        self.nodes[node.key] = node
        return node

    def directories(self, root, namespace):
        base = safe_file(self.roots[root], namespace)
        if not base.exists():
            return []
        paths = sorted(base.iterdir())
        for path in paths:
            if path.is_symlink() or not path.is_dir() or not HEX.fullmatch(path.name):
                raise IntegrityError(f"Unknown managed generation: {path}")
        return paths

    def gate(self):
        from .library import _read_inputs
        from .library_reports import load_research
        from .relative_scores import load_selected_library

        data, root = self.roots["setup"], self.roots["library"]
        for base in (data, root):
            if (base / ".cluster-path-migration.json").exists():
                raise IntegrityError("Recover cluster-path migration before cleanup")
            if (base / ".staging").exists() and any((base / ".staging").iterdir()):
                raise IntegrityError("Nonempty staging tree blocks cleanup")
        for role in ("setup", "library"):
            selected_meta = self.metadata(role, "selection.json")
            paths = (
                ("prepared", "embedding")
                if role == "setup"
                else ("library", "relative")
            )
            for key in paths:
                self.metadata(role, selected_meta[key] + "/manifest.json")
        selected, _, _, pm, em, _, sets, _ = _read_inputs(data, False)
        pair = load_selected_library(root)
        s = self.metadata("library", "selection.json")
        if (
            s.get("preparation_id") != pm["id"]
            or s.get("embedding_id") != em["id"]
            or pair.relative is None
            or sorted(pair.library.signatures) != list(sets.cluster_ids)
        ):
            raise IntegrityError(
                "Cleanup needs a full selected setup/production/C/B pair; "
                "run a full build first"
            )
        self.metadata("setup", "selection.json")
        self.selected_preparation, self.selected_embedding = pm["id"], em["id"]
        self.selected_production = s["library_id"]
        self.selected_relative = s["relative_id"]
        self.selected_research = None
        if (root / "research-selection.json").exists():
            r = self.metadata("library", "research-selection.json")
            if (
                r["production_library_id"] != s["library_id"]
                or r["production_manifest_sha256"] != s["manifest_sha256"]
            ):
                raise IntegrityError("Selected research differs from production")
            m = load_research(safe_file(root, r["research"]))
            if (
                m["id"] != r["research_id"]
                or sha256(root / r["research"] / "manifest.json")
                != r["manifest_sha256"]
            ):
                raise IntegrityError("Selected research checksum mismatch")
            self.selected_research = m["id"]
        return selected

    def setup(self):
        from .cluster_sets import validate_sets
        from .embeddings import verify_embeddings
        from .setup_data import verify_prepared

        self.prep_values, self.embedding_values = {}, {}
        for path in self.directories("setup", "prepared"):
            relative = "prepared/" + path.name
            self.metadata("setup", relative + "/manifest.json")
            m, windows, members = verify_prepared(path, path.name)
            sets = validate_sets(path, m, windows, members)
            node = self.add(
                "setup", relative, "prepared", m, {"manifest.json", *m["files"]}
            )
            node.owners.add(path.name)
            self.prepared[path.name] = node
            self.prep_values[path.name] = (m, windows, sets)
        for path in self.directories("setup", "embeddings"):
            relative = "embeddings/" + path.name
            self.metadata("setup", relative + "/manifest.json")
            m = verify_embeddings(path)
            if m["id"] != path.name:
                raise IntegrityError("Embedding directory identity mismatch")
            self.metadata("setup", relative + "/manifest.json")
            node = self.add(
                "setup", relative, "embedding", m, {"manifest.json", *m["files"]}
            )
            self.embeddings[path.name] = node
            self.embedding_values[path.name] = m
        self.matrix_embeddings = defaultdict(set)
        self.inventory_preparations = defaultdict(set)
        for p, (pm, _, _) in self.prep_values.items():
            self.inventory_preparations[pm["inventory_id"]].add(p)
        for e, em in self.embedding_values.items():
            self.matrix_embeddings[em["files"]["embeddings.npy"]["sha256"]].add(e)
        for node in (
            self.prepared[self.selected_preparation],
            self.embeddings[self.selected_embedding],
        ):
            node.protected = "selected setup input"

    def check_binding(self, row, cluster, fold=None):
        p, e = row["prepared_id"], row["embedding_id"]
        if p not in self.prep_values or e not in self.embedding_values:
            raise IntegrityError(
                "Missing prepared/embedding owner; cannot prove retirement"
            )
        pm, _, sets = self.prep_values[p]
        em = self.embedding_values[e]
        v = row["inputs"]
        from .library_runtime import representation_contract
        from .setup_data import read_tsv, reference_identity

        cids = list(sets.cluster_members(cluster)) if cluster in sets.clusters else []
        if fold is not None:
            if p not in self.fold_maps:
                maps = defaultdict(dict)
                for item in read_tsv(
                    self.roots["setup"] / self.prepared[p].path / "folds.tsv"
                ):
                    maps[item["cluster_id"]][item["window_id"]] = int(item["fold_id"])
                self.fold_maps[p] = maps
            folds = self.fold_maps[p][cluster]
            validation = [w for w in cids if folds[w] == fold]
            if (
                v.get("fold_id") != fold
                or v.get("V_n") != len(validation)
                or v.get("V_ids_sha256") != identity(validation)
                or v.get("partition") != identity([[w, folds[w]] for w in cids])
            ):
                raise IntegrityError("Research partition ownership mismatch")
            cids = [w for w in cids if folds[w] != fold]
        if "reference" in v and v["reference"] != reference_identity(pm["reference"]):
            raise IntegrityError("Fit reference ownership mismatch")

        if (
            cluster not in sets.clusters
            or em["inventory_id"] != pm["inventory_id"]
            or v["cluster_id"] != cluster
            or v["C_ids_sha256"] != identity(cids)
            or v["C_n"] != len(cids)
            or v["B_ids_sha256"] != sets.summary(cluster)["background_ids_sha256"]
            or v["B_n"] != sets.summary(cluster)["background_count"]
            or v["matrix_sha256"] != em["files"]["embeddings.npy"]["sha256"]
            or v["row_map_sha256"] != em["files"]["rows.tsv"]["sha256"]
            or v["representation_contract_id"]
            != identity(representation_contract(em["identity_inputs"]["contract"]))
        ):
            raise IntegrityError("Owner does not match persisted fit inputs")
        return p, e

    def published(self):
        from .library import load_library
        from .library_reports import load_research
        from .relative_scores import load_relative_scores

        base = self.roots["library"]
        libraries = {}
        for path in sorted(base.iterdir()):
            if not HEX.fullmatch(path.name):
                continue
            m = self.metadata("library", path.name + "/manifest.json")
            if m["schema"] not in {"pclass-dark.library.v1", "pclass-dark.library.v2"}:
                raise IntegrityError("Unknown production schema blocks cleanup")
            library = load_library(path)
            node = self.add(
                "library", path.name, "production", m, {"manifest.json", *m["files"]}
            )
            self.production[path.name], libraries[path.name] = node, library
            if m["schema"].endswith(".v1"):
                node.protected = "legacy production retained"
                self.protect_recipe(node, m["identity_inputs"])
        for path in self.directories("library", "relative-scores"):
            relative = "relative-scores/" + path.name
            m = self.metadata("library", relative + "/manifest.json")
            if m["schema"] != "pclass-dark.relative-score-library.v1":
                raise IntegrityError("Unknown C/B schema")
            pid = m["source_production"]["id"]
            if pid not in libraries:
                raise IntegrityError("Companion has no verifiable production owner")
            rel = load_relative_scores(path, library=libraries[pid])
            if rel.relative_id != path.name:
                raise IntegrityError("Relative directory identity mismatch")
            node = self.add(
                "library", relative, "relative", m, {"manifest.json", *m["files"]}
            )
            targets = m["identity_inputs"]["targets"]
            owners = set()
            for row in targets:
                cluster = row["cluster_id"]
                owners.add(self.check_binding(row, cluster))
                self.fit_owners[row["fit_id"]].add(
                    (row["prepared_id"], row["embedding_id"])
                )
            if len(owners) != 1:
                raise IntegrityError("Mixed preparation ownership blocks retirement")
            p, e = owners.pop()
            production = self.production[pid]
            if production.owners and production.owners != {p}:
                raise IntegrityError("Conflicting production owners")
            production.owners.add(p)
            production.dependencies.update(
                (self.prepared[p].key, self.embeddings[e].key, node.key)
            )
            node.owners.add(p)
            node.dependencies.update(
                (production.key, self.prepared[p].key, self.embeddings[e].key)
            )
            self.companions[path.name] = node
        for pid, node in self.production.items():
            if not node.owners and not node.protected:
                # Unpaired v2 outputs cannot be assigned from filenames or counts.
                raise IntegrityError(
                    f"Unpaired production {pid}; ownership is unbounded"
                )
            if pid == self.selected_production:
                node.protected = "selected production"
        for path in self.directories("library", "research"):
            m = load_research(path)
            relative = "research/" + path.name
            self.metadata("library", relative + "/manifest.json")
            source = m["source_production"]
            pid = source["id"]
            if (
                pid not in self.production
                or sha256(base / pid / "manifest.json") != source["manifest_sha256"]
            ):
                raise IntegrityError("Research source production missing or changed")
            node = self.add(
                "library", relative, "research", m, {"manifest.json", *m["files"]}
            )
            node.owners.update(self.production[pid].owners)
            node.dependencies.add(self.production[pid].key)
            node.dependencies.update(self.production[pid].dependencies)
            self.research[path.name] = node
            if path.name == self.selected_research:
                node.protected = "selected research"
        for node in self.nodes.values():
            if self.selected_preparation in node.owners:
                node.protected = node.protected or "current-preparation experiment"

    def protect_recipe(self, node, recipe):
        for name, mapping in (
            ("prepared_id", self.prepared),
            ("embedding_id", self.embeddings),
        ):
            if recipe.get(name) not in mapping:
                raise IntegrityError("Unbounded legacy/archive dependency")
            node.dependencies.add(mapping[recipe[name]].key)

    def receipt(self, path):
        """Pure checkpoint checks; never instantiate a mutating checkpoint store."""
        m = read_metadata(path)
        if (
            m["schema"] != "pclass-dark.library-checkpoint.v1"
            or m["payload_id"] != identity(m["payload"])
            or set(m["files"]) != {a["file"] for a in m["arrays"].values()}
        ):
            raise IntegrityError("Invalid checkpoint receipt")
        if m["files"]:
            verify_files(path.parent, m["files"])
        for meta in m["arrays"].values():
            a = np.load(
                safe_file(path.parent, meta["file"]), mmap_mode="r", allow_pickle=False
            )
            if (
                list(a.shape) != meta["shape"]
                or a.dtype.str != meta["dtype"]
                or a.dtype.str != "<f8"
            ):
                raise IntegrityError("Invalid checkpoint array")
            flat = a.reshape(-1)
            for begin in range(0, flat.size, 1048576):
                if not np.isfinite(flat[begin : begin + 1048576]).all():
                    raise IntegrityError("Nonfinite checkpoint array")
        return m

    def fit(self, path, fit_id, inputs, fit_name="final"):
        from .fit_records import validate_result
        from .signatures import SELECTION_VERSION, _validate_profiles

        m = self.receipt(path / "complete.json")
        task = identity([fit_id, inputs["cluster_id"], fit_name, inputs])
        complete = identity([task, SELECTION_VERSION])
        result = m["payload"]
        sig = result["signature"]
        if (
            m["task_id"] != task
            or m["dependency"] != complete
            or sig["input_identity"] != inputs
            or sig["library_id"] != fit_id
            or sig["counts"]["training_positive_n"] != inputs["C_n"]
            or sig["counts"]["background_n"] != inputs["B_n"]
            or sig["counts"]["validation_positive_n"] != inputs.get("V_n", 0)
            or (
                "V_ids_sha256" in inputs
                and identity([r["window_id"] for r in result["oof"]])
                != inputs["V_ids_sha256"]
            )
            or m["arrays"]
            or m["files"]
        ):
            raise IntegrityError("Completed fit identity mismatch")
        validate_result(
            result,
            cluster=inputs["cluster_id"],
            fit=fit_name,
            counts=sig["counts"],
            sigma_floor=inputs["sigma_floor"],
            validation_ids=[r["window_id"] for r in result["oof"]],
        )
        expected = {"complete.json"}
        trace = result["trace"]
        params = identity(
            [
                [p["mu_C"], p["sigma_C"], p["effective_sigma"]]
                for p in result["profiles"]
            ]
        )
        selection_dependency = identity(
            [SELECTION_VERSION, identity([complete, params, trace["order"]])]
        )
        selection = trace["selection"]
        decisions = selection.get("iterations", [])
        decision_dependencies = [
            identity(
                [
                    selection_dependency,
                    d["before"],
                    selection["core"],
                    d["before_count"],
                ]
            )
            for d in decisions
        ]
        # Complete fits normally have only profile receipts after numeric cleanup.
        for file in path.iterdir():
            if file.name == "initial.json":
                r = self.receipt(file)
                if (
                    r["task_id"] != task
                    or r["dependency"]
                    != identity([selection_dependency, selection["candidates"]])
                    or r["payload"] != {"count": selection["initial_count"]}
                ):
                    raise IntegrityError("Initial selection receipt mismatch")
                expected.add(file.name)
            elif re.fullmatch(r"decision-\d{4}\.json", file.name):
                index = int(file.stem.split("-")[1])
                r = self.receipt(file)
                if (
                    index >= len(decisions)
                    or r["task_id"] != task
                    or r["dependency"] != decision_dependencies[index]
                    or r["payload"] != decisions[index]
                ):
                    raise IntegrityError("Selection decision receipt mismatch")
                expected.add(file.name)
            elif re.fullmatch(r"trial-\d{4}-\d{3}\.json", file.name):
                _, number, dimension = file.stem.split("-")
                i, d = int(number), int(dimension)
                r = self.receipt(file)
                trials = decisions[i]["trials"] if i < len(decisions) else []
                trial = next((t for t in trials if t["dimension"] == d), None)
                if (
                    trial is None
                    or r["task_id"] != task
                    or r["dependency"] != identity([decision_dependencies[i], d])
                    or r["payload"] != {"count": trial["count"]}
                ):
                    raise IntegrityError("Removal trial receipt mismatch")
                expected.add(file.name)
            elif file.name == "source.json":
                read_metadata(file)
                expected.add(file.name)
            elif re.fullmatch(r"profiles-\d{4}\.json", file.name):
                end = int(file.stem.split("-")[1])
                r = self.receipt(file)
                if (
                    end % 16
                    or not 16 <= end <= 640
                    or r["task_id"] != task
                    or r["dependency"] != identity([complete, end - 16, end])
                ):
                    raise IntegrityError("Profile checkpoint identity mismatch")
                _validate_profiles(
                    r["payload"]["rows"],
                    range(end - 16, end),
                    inputs["C_n"],
                    inputs["B_n"],
                    inputs.get("V_n", 0),
                    inputs["sigma_floor"],
                )
                expected.add(file.name)
        return m, expected

    def checkpoints(self):
        base = self.roots["library"]
        for path in self.directories("library", ".fits"):
            relative = ".fits/" + path.name
            id_path = path / "identity.json"
            if not id_path.exists():
                raise IntegrityError("Unidentified fit checkpoint blocks cleanup")
            m = self.metadata("library", relative + "/identity.json")
            inputs = m["identity_inputs"]
            if (
                m["schema"] != "pclass-dark.fit-inputs.v1"
                or identity(inputs) != path.name
                or m["id"] != path.name
            ):
                raise IntegrityError("Invalid fit identity record")
            owners = self.fit_owners.get(path.name, set())
            if not owners:
                # Bind partial work conservatively to all compatible matrix inventories.
                owners = {
                    (p, e)
                    for e in self.matrix_embeddings[inputs["matrix_sha256"]]
                    for p in self.inventory_preparations[
                        self.embedding_values[e]["inventory_id"]
                    ]
                }
            if not owners:
                raise IntegrityError("Fit checkpoint has unbounded owners")
            expected = None
            protected = None
            if (path / "final/complete.json").exists():
                _, files = self.fit(path / "final", path.name, inputs)
                expected = {"identity.json", *("final/" + f for f in files)}
            else:
                protected = "partial resumable fit"
            node = self.add("library", relative, "fit", m, expected)
            node.protected = protected or node.protected
            for p, e in owners:
                node.owners.add(p)
                node.dependencies.update((self.prepared[p].key, self.embeddings[e].key))
            self.fit_nodes[path.name] = node
        for namespace, kind in (
            (".production-checkpoints", "production-checkpoints"),
            (".research-checkpoints", "research-checkpoints"),
        ):
            for path in self.directories("library", namespace):
                phase = (
                    self.production.get(path.name)
                    if kind.startswith("production")
                    else self.research.get(path.name)
                )
                if phase is None:
                    records = list(path.rglob("target.json")) + list(
                        path.rglob("fold_*-target.json")
                    )
                    partial = self.add(
                        "library", namespace + "/" + path.name, kind, {"id": path.name}
                    )
                    partial.protected = "unpublished resumable checkpoint run"
                    for file in records:
                        t = read_metadata(file)
                        if t.get("schema") == "pclass-dark.research-target.v1":
                            continue
                        if t.get("schema") != "pclass-dark.target-result.v2":
                            raise IntegrityError("Unknown partial run schema")
                        self.check_target_owner(t)
                        self.retain_ids(partial, t["prepared_id"], t["embedding_id"])
                        partial.owners.add(t["prepared_id"])
                    if not partial.owners:
                        raise IntegrityError("Unidentified partial run blocks cleanup")
                    partial.dependencies.update(
                        n.key
                        for n in self.fit_nodes.values()
                        if n.owners & partial.owners
                    )
                    continue
                node = self.add(
                    "library", namespace + "/" + path.name, kind, {"id": path.name}
                )
                node.owners.update(phase.owners)
                node.dependencies.update(phase.dependencies)
                expected = set()
                targets = phase.metadata["identity_inputs"]["cluster_ids"]
                from .cluster_paths import cluster_directory_name

                complete = True
                for cluster in targets:
                    directory = cluster_directory_name(cluster)
                    # Current private checkpoint paths are encoded, unlike public paths.
                    encoded = "c_" + cluster.encode().hex()
                    if (path / encoded).exists():
                        directory = encoded
                    prefix = directory + "/"
                    fits = (
                        ["final"]
                        if kind.startswith("production")
                        else [f"fold_{f}" for f in range(5)]
                    )
                    for fit_name in fits:
                        target_file = (
                            path
                            / prefix
                            / (
                                "target.json"
                                if fit_name == "final"
                                else fit_name + "-target.json"
                            )
                        )
                        if not target_file.exists():
                            complete = False
                            continue
                        t = read_metadata(target_file)
                        if (
                            t["schema"] != "pclass-dark.target-result.v2"
                            or t["status"] != "complete"
                            or t["cluster_id"] != cluster
                        ):
                            raise IntegrityError("Invalid target receipt")
                        result_path = safe_file(base, t["result"])
                        if not result_path.exists():
                            # A completed publication is self-contained, but a
                            # missing private receipt means its cache must resume.
                            complete = False
                            self.retain_ids(node, t["prepared_id"], t["embedding_id"])
                            if fit_name == "final":
                                self.check_binding(
                                    {
                                        "prepared_id": t["prepared_id"],
                                        "embedding_id": t["embedding_id"],
                                        "inputs": t["fit_inputs"],
                                    },
                                    cluster,
                                )
                            continue
                        if sha256(result_path) != t["result_sha256"]:
                            raise IntegrityError("Target points to a changed fit")
                        self.check_target_owner(t)
                        if (
                            t["prepared_id"] not in phase.owners
                            or self.embeddings[t["embedding_id"]].key
                            not in phase.dependencies
                        ):
                            raise IntegrityError(
                                "Conflicting published/target ownership"
                            )
                        if fit_name == "final":
                            fit_node = self.fit_nodes.get(t["fit_id"])
                            if (
                                fit_node is None
                                or result_path
                                != base / fit_node.path / "final/complete.json"
                            ):
                                raise IntegrityError("Target fit path mismatch")
                            node.dependencies.add(fit_node.key)
                            expected.add(prefix + "target.json")
                        else:
                            _, files = self.fit(
                                result_path.parent,
                                t["fit_id"],
                                t["fit_inputs"],
                                fit_name,
                            )
                            expected.update(prefix + fit_name + "/" + f for f in files)
                            expected.add(prefix + fit_name + "-target.json")
                            expected.add(prefix + fit_name + "-identity.json")
                    if fits != ["final"]:
                        if not (path / prefix / "target.json").exists():
                            complete = False
                        else:
                            aggregate = read_metadata(path / prefix / "target.json")
                            paths = [
                                str(path / prefix / (f + "-target.json")) for f in fits
                            ]
                            if (
                                aggregate.get("schema")
                                != "pclass-dark.research-target.v1"
                                or aggregate.get("status") != "complete"
                                or aggregate.get("cluster_id") != cluster
                                or aggregate.get("required_fits") != fits
                            ):
                                raise IntegrityError(
                                    "Unknown research aggregate receipt"
                                )
                            if aggregate.get("fits") != paths:
                                complete = False
                            expected.add(prefix + "target.json")
                    background = path / prefix / "background.json"
                    if background.exists():
                        self.receipt(background)
                        expected.add(prefix + "background.json")
                if not complete:
                    node.protected = "partial resumable checkpoint run"
                    node.dependencies.update(
                        n.key for n in self.fit_nodes.values() if n.owners & node.owners
                    )
                else:
                    checked = self.add(
                        "library", node.path, kind, node.metadata, expected
                    )
                    checked.owners, checked.dependencies = (
                        node.owners,
                        node.dependencies,
                    )
        for path in self.directories("library", ".relative-checkpoints"):
            r = (
                self.receipt(path / "complete.json")
                if (path / "complete.json").exists()
                else None
            )
            fit = self.fit_nodes.get(path.name)
            owners = self.fit_owners.get(path.name, set())
            if fit is None and not owners:
                raise IntegrityError("Unowned relative checkpoint blocks cleanup")
            expected = {"complete.json", *r["files"]} if r else None
            node = self.add(
                "library",
                ".relative-checkpoints/" + path.name,
                "relative-checkpoint",
                {"id": path.name},
                expected,
            )
            if fit:
                node.dependencies.add(fit.key)
                node.owners.update(fit.owners)
            else:
                node.owners.update(p for p, _ in owners)
            if r is None:
                node.protected = "partial relative checkpoint"
            elif r["task_id"] != path.name or r["payload"]["fit_id"] != path.name:
                raise IntegrityError("Distribution checkpoint fit mismatch")
        # Published fits do not read checkpoints; shared IDs still protect reuse.
        for fit_id, owners in self.fit_owners.items():
            if any(p == self.selected_preparation for p, _ in owners):
                for key in (
                    "library:.fits/" + fit_id,
                    "library:.relative-checkpoints/" + fit_id,
                ):
                    if key in self.nodes:
                        self.nodes[
                            key
                        ].protected = "fit shared with current preparation"

    def check_target_owner(self, t):
        row = {
            "prepared_id": t["prepared_id"],
            "embedding_id": t["embedding_id"],
            "inputs": t["fit_inputs"],
        }
        required = t["required_fits"]
        if required == ["final"]:
            fold = None
        elif len(required) == 1 and re.fullmatch(r"fold_[0-4]", required[0]):
            fold = int(required[0][-1])
        else:
            raise IntegrityError("Unknown target fit scope")
        self.check_binding(row, t["cluster_id"], fold)
        if identity(t["fit_inputs"]) != t["fit_id"]:
            raise IntegrityError("Target normalized fit identity mismatch")
        result = self.receipt(safe_file(self.roots["library"], t["result"]))["payload"]
        from .fit_records import payload_identity

        if (
            payload_identity(result) != t["payload_id"]
            or result["signature"]["input_identity"] != t["fit_inputs"]
            or result["signature"]["cluster_id"] != t["cluster_id"]
            or result["signature"]["fit_id"] != required[0]
        ):
            raise IntegrityError("Target scientific payload mismatch")
        if "relative" in t:
            p = safe_file(self.roots["library"], t["relative"] + "/complete.json")
            r = self.receipt(p)
            if (
                sha256(p) != t["relative_receipt_sha256"]
                or r["payload"]["payload_id"] != t["payload_id"]
            ):
                raise IntegrityError("Target relative checkpoint mismatch")

    def shards(self):
        from .embeddings import SHARD_ROWS, _finite_matrix

        for path in self.directories("setup", ".checkpoints"):
            m = self.embedding_values.get(path.name)
            if m is None:
                receipts = sorted(path.glob("*.json"))
                partial = self.add(
                    "setup",
                    ".checkpoints/" + path.name,
                    "embedding-checkpoints",
                    {"id": path.name},
                )
                partial.protected = "partial resumable embedding job"
                inventories = set()
                for file in receipts:
                    r = read_metadata(file)
                    if (
                        r.get("schema") != "pclass-dark.embedding-shard.v1"
                        or r.get("embedding_id") != path.name
                        or identity(
                            {
                                "inventory_id": r["inventory_id"],
                                "contract": r["execution"]["contract"],
                            }
                        )
                        != path.name
                    ):
                        raise IntegrityError("Unknown partial embedding receipt")
                    inventories.add(r["inventory_id"])
                if len(inventories) != 1:
                    raise IntegrityError("Unbounded partial embedding inventory")
                for p, (pm, _, _) in self.prep_values.items():
                    if pm["inventory_id"] in inventories:
                        partial.dependencies.add(self.prepared[p].key)
                        partial.owners.add(p)
                if not partial.owners:
                    raise IntegrityError("Partial embedding has no bounded preparation")
                continue
            node = self.add(
                "setup",
                ".checkpoints/" + path.name,
                "embedding-checkpoints",
                {"id": path.name},
            )
            node.dependencies.add(self.embeddings[path.name].key)
            expected = set()
            complete = True
            for begin in range(0, m["rows"], SHARD_ROWS):
                stop = min(begin + SHARD_ROWS, m["rows"])
                name = f"{begin:012d}"
                a, b = path / (name + ".json"), path / (name + ".npy")
                if not a.exists() or not b.exists():
                    complete = False
                    continue
                r = read_metadata(a)
                from .artifacts import file_record

                if (
                    r["schema"] != "pclass-dark.embedding-shard.v1"
                    or r["embedding_id"] != path.name
                    or r["inventory_id"] != m["inventory_id"]
                    or r["start"] != begin
                    or r["stop"] != stop
                    or r["file"] != file_record(b)
                    or r["shape"] != [stop - begin, 640]
                    or r["dtype"] != "<f4"
                    or identity(
                        {
                            "inventory_id": r["inventory_id"],
                            "contract": r["execution"]["contract"],
                        }
                    )
                    != path.name
                ):
                    raise IntegrityError("Invalid embedding shard")
                array = np.load(b, mmap_mode="r", allow_pickle=False)
                _finite_matrix(array, stop - begin)
                if array.shape != (stop - begin, 640) or array.dtype.str != "<f4":
                    raise IntegrityError("Invalid shard shape")
                expected.update((a.name, b.name))
            if not complete:
                node.protected = "partial embedding checkpoint cache"
            else:
                checked = self.add(
                    "setup", node.path, node.kind, node.metadata, expected
                )
                checked.dependencies = node.dependencies
            if path.name == self.selected_embedding:
                self.nodes[node.key].protected = "selected embedding recovery shards"

    def retain_ids(self, node, p=None, e=None, production=None):
        for value, mapping in (
            (p, self.prepared),
            (e, self.embeddings),
            (production, self.production),
        ):
            if value is not None:
                if value not in mapping:
                    raise IntegrityError(
                        "Retained history references missing managed inputs"
                    )
                node.dependencies.add(mapping[value].key)

    def retained_metadata(self, node, m):
        schema = m.get("schema", "")
        if schema in {
            "pclass-dark.selection.v1",
            "pclass-dark.library-selection.v1",
            "pclass-dark.library-selection.v2",
        }:
            self.retain_ids(node, m.get("preparation_id"), m.get("embedding_id"))
            if m.get("library_id") in self.production:
                self.retain_ids(node, production=m["library_id"])
        elif schema == "pclass-dark.library-research-selection.v1":
            self.retain_ids(node, production=m["production_library_id"])
        elif schema == "pclass-dark.relative-score-library.v1":
            if identity(m["identity_inputs"]) != m["id"]:
                raise IntegrityError("Archived relative identity mismatch")
            for row in m["identity_inputs"]["targets"]:
                self.retain_ids(node, row["prepared_id"], row["embedding_id"])
        elif schema == "pclass-dark.library.v1":
            self.protect_recipe(node, m["identity_inputs"])
        elif schema == "pclass-dark.library.v2":
            if identity(m["identity_inputs"]) != m["id"]:
                raise IntegrityError("Archived production identity mismatch")
            owners = set()
            for _, fit_id in m["identity_inputs"]["final_fits"]:
                bound = self.fit_owners.get(fit_id, set())
                if not bound:
                    raise IntegrityError("Unbounded archived fit dependencies")
                owners.update(bound)
            if not owners:
                raise IntegrityError("Unbounded archived production dependencies")
            for p, e in owners:
                self.retain_ids(node, p, e)
        elif schema == "pclass-dark.library-research.v1":
            self.retain_ids(node, production=m["source_production"]["id"])
        elif schema in {"pclass-dark.prepared.v1", "pclass-dark.embeddings.v1"}:
            # Self-contained archive. Embedding origin is provenance only.
            pass
        elif "signature" in m:
            sig = m["signature"]
            from .library_runtime import validate_signature

            validate_signature(sig)
            owners = self.fit_owners.get(
                sig.get("fit_identity", identity(sig["input_identity"])), set()
            )
            if owners:
                for p, e in owners:
                    self.retain_ids(node, p, e)
            elif "prepared_id" in sig["input_identity"]:
                self.protect_recipe(node, sig["input_identity"])
            else:
                raise IntegrityError("Unbounded retained migration source")
        elif schema == "pclass-dark.library-checkpoint.v1":
            self.retained_metadata(node, m["payload"])
        elif "fit_id" in m and "counts" in m and "payload_id" in m:
            owners = self.fit_owners.get(m["fit_id"], set())
            if not owners:
                raise IntegrityError("Unbounded archived distribution checkpoint")
            for p, e in owners:
                self.retain_ids(node, p, e)
        else:
            raise IntegrityError("Unknown retained-history schema blocks cleanup")

    def archives(self):
        for role, root in self.roots.items():
            for namespace in (".archives", ".migration", ".checkpoints"):
                if namespace == ".checkpoints" and role == "setup":
                    continue
                base = safe_file(root, namespace)
                if not base.exists() or not any(base.iterdir()):
                    continue
                node = self.add(role, namespace, "retained-history", {"id": namespace})
                node.protected = "explicit archive or legacy/migration cache"
                if node.tree is None:
                    raise IntegrityError(
                        "Unsafe retained history blocks dependency analysis"
                    )
                records = set(base.rglob("manifest.json"))
                records.update(
                    p
                    for p in base.rglob("*")
                    if p.is_file()
                    and p.name.startswith(("selection.json", "research-selection"))
                )
                records.update(base.rglob("complete.json"))
                if namespace in {".migration", ".checkpoints"}:
                    records.update(base.rglob("final.json"))
                if not records:
                    raise IntegrityError(
                        "Retained history has no bounded ownership metadata"
                    )
                covered = set()
                for path in sorted(records):
                    m = self.metadata(role, path.relative_to(root).as_posix())
                    if "manifest.json" == path.name:
                        verify_files(path.parent, m["files"])
                        covered.update(path.parent / name for name in m["files"])
                    if path.name == "complete.json":
                        m = self.receipt(path)
                    self.retained_metadata(node, m)
                    covered.add(path)
                # Loose unrecognized files may hold unbounded dependency records.
                for file in base.iterdir():
                    if file.is_file() and file not in covered:
                        raise IntegrityError("Unrecognized loose retained-history file")
                if namespace == ".archives":
                    # The full selected replacement passed gate(); these are
                    # fallback copies, not permanent roots of the retention graph.
                    node.kind = "rollback-history"
                    node.protected = None

    def finish(self):
        # Any retained checkpoint/publication pins all mandatory dependencies.
        live = {n.key for n in self.nodes.values() if n.protected}
        changed = True
        while changed:
            before = len(live)
            for key in tuple(live):
                live.update(self.nodes[key].dependencies)
            changed = len(live) != before
        obsolete = {p for p, n in self.prepared.items() if n.key not in live}
        for node in self.nodes.values():
            if node.kind == "embedding":
                eid = node.metadata["id"]
                consumers = {
                    p
                    for owners in self.fit_owners.values()
                    for p, e in owners
                    if e == eid
                }
                origin = node.metadata.get("originating_preparation_id")
                if origin in self.prepared:
                    consumers.add(origin)
                if eid == self.selected_embedding:
                    consumers.add(self.selected_preparation)
                node.owners.update(consumers)
            elif node.kind == "embedding-checkpoints":
                embedding = self.embeddings.get(node.metadata["id"])
                if embedding is not None:
                    node.owners.update(embedding.owners)
        # Preserve every owner not retired, then close required dependencies again.
        live.update(
            n.key
            for n in self.nodes.values()
            if n.kind != "rollback-history"
            and (not n.owners or not n.owners <= obsolete)
        )
        changed = True
        while changed:
            before = len(live)
            for key in tuple(live):
                live.update(self.nodes[key].dependencies)
            changed = len(live) != before
        entries, inventories = [], InventoryStore()
        try:
            for node in sorted(self.nodes.values(), key=lambda n: n.key):
                tree = node.tree
                remove = node.key not in live
                files = tree["files"] if tree else []
                logical = sum(f["size_bytes"] for f in files)
                if remove:
                    inventories[node.key] = tree
                entries.append(
                    CleanupEntry(
                        node.root,
                        node.path,
                        node.kind,
                        node.metadata.get("id", Path(node.path).name),
                        "remove" if remove else "protect",
                        (
                            "superseded rollback history"
                            if node.kind == "rollback-history"
                            else "obsolete exclusive generation"
                        )
                        if remove
                        else node.protected or "required by retained artifact",
                        tuple(sorted(node.dependencies)),
                        identity(tree) if tree else None,
                        len(files),
                        logical,
                        None,
                    )
                )
            from dataclasses import replace

            valued = [
                replace(
                    e,
                    allocated_bytes_estimate=(
                        inventories.allocated(inventories[e.key])
                        if e.disposition == "remove"
                        else 0
                    ),
                )
                for e in entries
            ]
            snapshot = tuple((r, p, h) for (r, p), h in sorted(self.snapshots.items()))
            digest = identity(
                {
                    "inventories": {
                        e.key: e.tree_sha256
                        for e in entries
                        if e.disposition == "remove"
                    },
                    "owners": {n.key: sorted(n.owners) for n in self.nodes.values()},
                }
            )
            value = CleanupPlan(
                "pclass-dark.cleanup-plan.v1",
                str(self.roots["setup"]),
                str(self.roots["library"]),
                snapshot,
                tuple(valued),
                digest,
                "",
            )
            d = value.to_dict()
            d.pop("plan_id")
            from dataclasses import replace

            return replace(value, plan_id=identity(d)), inventories
        except BaseException:
            inventories.close()
            raise


def make_plan(data, library):
    planner = Planner(data, library)
    try:
        for phase in (
            "gate",
            "setup",
            "published",
            "checkpoints",
            "shards",
            "archives",
        ):
            LOG.info("Cleanup: validating %s", phase)
            getattr(planner, phase)()
        result = planner.finish()
        LOG.info(
            "Cleanup: planned %s removable generations",
            sum(e.disposition == "remove" for e in result[0].entries),
        )
        return result
    finally:
        planner.spool.close()
