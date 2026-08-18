"""``graph_edge_semantics`` — graph traversal under hidden edge semantics.

Mechanism: a small labelled digraph is displayed in full, but what each edge
LABEL means is hidden.  Each of the 5-6 labels is secretly one of
``traversable`` (u -> v), ``blocked`` (no traversal), ``reversed`` (v -> u) or
``bidirectional`` (both).  The task is reachability between two named nodes.
Answer: ``YES``/``NO``.

Difficulty 0/1/2 -> 5/6/7 nodes and 5/6/8 edges.

TRANSFER SHIFT: a larger graph (7 nodes, 9 edges) whose queries need
longer traversals than any training instance, so a learner that memorised short
one- or two-hop patterns under the hidden semantics does not carry over.  The
latent label semantics is unchanged.

Structured feedback: the effective out-degree (under the hidden semantics) of
ONE displayed node, identified by its opaque index in the displayed node list.
The reachability bit itself is never stated.

WHICH node is hinted is chosen ANSWER-INDEPENDENTLY (generator 1.1.0)
------------------------------------------------------------------
Generator 1.0.0 drew the hinted node from the queried PATH when one existed and
used the source otherwise (contract §4's "one named node on the queried path").
That made the SELECTION itself a decode channel: "the hinted node is not the
source" and "the hinted node is the target" are features a reader can compute
from the prompt alone, and both were deterministic evidence for ``YES``
(measured P = 1.0 at n = 382 / n = 291 over 2160 sampled items).  The
reachability bit was therefore readable without ever inferring the hidden edge
semantics.

Generator 1.1.0 selects the hinted node from the FULL displayed node list with
a generator seeded ONLY by displayed data (edges, source, target, node count),
i.e. by nothing that depends on the hidden semantics.  The selection channel
consequently carries no reachability information (measured: every
model-computable branch within 0.05 of the base rate), while the hint content
stays exactly as informative as §4 requires — the TRUE out-degree of a named
node under the hidden semantics.  Recorded in Amendment 13.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, Rule, TaskFamily,
                    TaskInstance, derive_seed, letters, make_rng)
from ..surface_remap import PromptSpec, lab, sym

NODE_POOL = ("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9")
LABEL_POOL = ("lx", "ly", "lz", "lw", "lv", "lu")
SYMBOL_POOL = NODE_POOL + LABEL_POOL
SEMANTICS = ("traversable", "blocked", "reversed", "bidirectional")
_SIZES = {0: (5, 5), 1: (6, 6), 2: (7, 8)}
_TRANSFER_SIZE = (7, 9)


def _adjacency(edges, semantics, n_nodes: int) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {i: set() for i in range(n_nodes)}
    for u, v, label in edges:
        meaning = semantics[label]
        if meaning in ("traversable", "bidirectional"):
            adj[u].add(v)
        if meaning in ("reversed", "bidirectional"):
            adj[v].add(u)
    return adj


def _bfs(adj, source: int, target: int) -> list[int] | None:
    seen = {source: None}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            path = [node]
            while seen[path[-1]] is not None:
                path.append(seen[path[-1]])
            return list(reversed(path))
        for nxt in sorted(adj[node]):
            if nxt not in seen:
                seen[nxt] = node
                queue.append(nxt)
    return None


class GraphEdgeSemanticsFamily(TaskFamily):
    family_id = "graph_edge_semantics"
    #: 1.0.0 -> 1.1.0: answer-independent hint-node selection (Amendment 13).
    #: The bump changes rule_id / instance_id / episode ids for this family and
    #: the family-split manifest's generator source hash; the data is
    #: regenerated, the SPLIT itself is unchanged (family ids are unchanged).
    generator_version = "1.1.0"
    canon_mode = "label"
    symbol_pool = SYMBOL_POOL
    symbol_blocks = (NODE_POOL, LABEL_POOL)
    label_variants = (("TRUE", "FALSE"), ("REACHABLE", "UNREACHABLE"),
                      ("OPEN", "SHUT"))

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        n_labels = int(rng.integers(5, 7))
        for _ in range(64):
            meanings = [SEMANTICS[int(i)] for i in
                        rng.integers(0, len(SEMANTICS), size=n_labels)]
            if any(m in ("traversable", "bidirectional") for m in meanings):
                break
        return Rule(self.family_id, {"n_labels": n_labels, "semantics": meanings})

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        n_labels = rule.params["n_labels"]
        semantics = rule.params["semantics"]
        n_nodes, n_edges = _TRANSFER_SIZE if kind == KIND_TRANSFER \
            else _SIZES[difficulty]

        edges = []
        for _ in range(n_edges):
            u, v = (int(x) for x in rng.choice(n_nodes, size=2, replace=False))
            edges.append((u, v, int(rng.integers(0, n_labels))))
        adj = _adjacency(edges, semantics, n_nodes)

        source = int(rng.integers(0, n_nodes))
        reachable = [t for t in range(n_nodes)
                     if t != source and _bfs(adj, source, t) is not None]
        unreachable = [t for t in range(n_nodes)
                       if t != source and t not in reachable]
        prefer_yes = int(rng.integers(0, 2)) == 0
        pool = (reachable if prefer_yes else unreachable) or \
            (unreachable if prefer_yes else reachable)
        target = int(rng.choice(pool)) if pool else (source + 1) % n_nodes
        path = _bfs(adj, source, target)
        answer = "YES" if path is not None else "NO"

        node_line = ", ".join(sym(NODE_POOL[i]) for i in range(n_nodes))
        edge_lines = ", ".join(
            f"{sym(NODE_POOL[u])}-{sym(LABEL_POOL[label])}>{sym(NODE_POOL[v])}"
            for u, v, label in edges)
        clauses = [
            "Each edge below carries a label whose meaning is hidden and fixed.",
            f"Nodes: {node_line}",
            f"Edges: {edge_lines}",
        ]
        spec = PromptSpec(
            clauses=tuple(clauses),
            question=(f"Can {sym(NODE_POOL[target])} be reached from "
                      f"{sym(NODE_POOL[source])}?"),
            answer_format=(f"Reply with a final line: ANSWER: {lab('YES')} "
                           f"or {lab('NO')}"),
            alphabet=tuple(NODE_POOL[:n_nodes]) + tuple(LABEL_POOL[:n_labels]),
            labels=("YES", "NO"),
            answer=lab(answer),
        )
        return Item(spec=spec, data={"edges": edges, "source": source,
                                     "target": target, "n_nodes": n_nodes,
                                     "path": path, "answer": answer})

    # -- feedback -----------------------------------------------------------

    def _hint_node(self, item: Item) -> int:
        """The hinted node: uniform over the DISPLAYED nodes, answer-independent.

        The seed is derived from DISPLAYED data only (the edge list, the queried
        source and target, the node count).  The hidden label semantics — the
        only thing reachability depends on — is deliberately absent from the
        seed, so the choice of node cannot carry one bit about the answer.  It
        is still a pure function of the instance, so the same item always gets
        the same hinted node no matter which attempt the hint answers.
        """
        data = item.data
        seed = derive_seed(0, self.family_id, "HINT_NODE",
                           [[int(u), int(v), int(label)]
                            for u, v, label in data["edges"]],
                           int(data["source"]), int(data["target"]),
                           int(data["n_nodes"]))
        return int(make_rng(seed).integers(0, int(data["n_nodes"])))

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        n_nodes = item.data["n_nodes"]
        adj = _adjacency(item.data["edges"], rule.params["semantics"], n_nodes)
        node = self._hint_node(item)
        degree = len(adj[node])
        names = tuple(letters(i) for i in range(n_nodes))
        return Feedback("HINT-DEGREE", (
            HintField("at", "PT", node, n_nodes, names),
            HintField("outdeg", "DEG", degree, n_nodes + 1),
        ))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        return self._flip_label(rule, instance)


FAMILY = GraphEdgeSemanticsFamily()
