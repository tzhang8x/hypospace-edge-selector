"""HypoSpace: verified edge selection for repeated Qwen answers on causal DAGs.

Python 3.10+, dependencies: python -m pip install numpy networkx
Offline reproduction: python hypospace.py --output results
Optional fresh API baseline (uses tokens): python hypospace.py --collect 5 --output new-run
Then replay: python hypospace.py --output new-run
For a fresh four-node baseline use --collect 4. Each run takes twelve draws per question.
Existing completed calls in the output directory are reused. The default run uses the
recorded Qwen3.8-27B graphs. It trains on four nodes and evaluates frozen weights on five.
The embedded records contain only observation masks and parsed graph masks, no credentials.

Benchmark: https://github.com/CTT-Pavilion/_HypoSpace
Paper: https://arxiv.org/abs/2510.15614
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
from itertools import combinations, permutations
from pathlib import Path
from urllib.request import Request, urlopen

import networkx as nx
import numpy as np

SEED = 41
MODEL = "qwen3.8-27b"
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def observation_family(observations, nodes):
    """Keep instances equivalent under node relabeling in the same split."""
    return min(
        tuple(sorted((labels[obs["perturbed_node"]], tuple(sorted(
            (labels[node], effect) for node, effect in obs["effects"].items()
        ))) for obs in observations))
        for order in permutations(nodes)
        for labels in [dict(zip(nodes, order))]
    )



class Episode:
    def __init__(self, item, rows, nodes):
        self.nodes = tuple(nodes)
        self.edges = tuple(permutations(nodes, 2))
        self.name = item["observation_set_id"]
        self.obs = item["observations"]
        self.family = observation_family(self.obs, self.nodes)
        self.truth = {frozenset(map(tuple, g["edges"])) for g in item["ground_truth_graphs"]}
        self.draws = [frozenset(map(tuple, r["edges"])) if r["valid"] else None for r in rows]
        for row in rows:
            graph = row["edges"]
            assert row["valid"] == (graph is not None and frozenset(map(tuple, graph)) in self.truth)
        self.effects = {(o["perturbed_node"], t): v for o in self.obs for t,v in o["effects"].items()}

    @lru_cache(None)
    def verify(self, edges):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(edges)
        return nx.is_directed_acyclic_graph(graph) and all(
            nx.descendants(graph, obs["perturbed_node"]) ==
            {n for n, effect in obs["effects"].items() if effect}
            for obs in self.obs
        )


class TransferEpisode(Episode):
    @lru_cache(None)
    def structure(self, graph):
        current = nx.DiGraph()
        current.add_nodes_from(self.nodes)
        current.add_edges_from(graph)
        reach = {node: nx.descendants(current, node) for node in self.nodes}
        return current, reach

    @lru_cache(None)
    def features(self, graph, edit):
        # All features refer to the current graph. Candidate validation is budgeted separately.
        source, target = edit
        current, reach = self.structure(graph)
        values = [edit in graph, (target, source) in graph,
                  target in reach[source], source in reach[target],
                  self.effects.get(edit, -1), self.effects.get((target, source), -1)]
        scale = len(self.nodes) - 1
        for node in edit:
            values.extend((current.in_degree(node)/scale, current.out_degree(node)/scale,
                           len(reach[node])/scale, sum(node in r for r in reach.values())/scale))
        intermediates = sum((source, node) in graph and target in reach[node]
                            for node in self.nodes if node not in edit)
        values.append(intermediates / (scale-1))
        # Relative positions of the edited endpoints in observed perturbation responses.
        patterns = [0]*9
        for obs in self.obs:
            def state(node):
                return 2 if node == obs["perturbed_node"] else obs["effects"][node]
            patterns[3*state(source) + state(target)] += 1
        return tuple(values + patterns)



def fit(episodes):
    examples = {}
    for ep in episodes:
        for graph, count in Counter(ep.draws).items():
            if graph is None or count < 2:
                continue
            for edit in ep.edges:
                x = ep.features(graph, edit)
                label = ep.verify(graph ^ {edit})
                examples.setdefault(x, []).append(label)
    x = np.asarray(list(examples), dtype=float)
    y = np.asarray([np.mean(labels) for labels in examples.values()])[:, None]
    rng = np.random.default_rng(SEED)
    params = [rng.normal(0, 0.2, (x.shape[1], 32)), np.zeros(32),
              rng.normal(0, 0.2, (32, 1)), np.zeros(1)]
    m, v = ([np.zeros_like(p) for p in params] for _ in range(2))
    for step in range(1, 801):
        idx = rng.integers(len(y), size=64)
        batch = x[idx]
        w, b, out, bias = params
        hidden = np.tanh(batch @ w + b)
        delta = (1 / (1 + np.exp(-np.clip(hidden @ out + bias, -40, 40))) - y[idx]) / len(idx)
        back = (delta @ out.T) * (1 - hidden**2)
        grads = [batch.T @ back, back.sum(0), hidden.T @ delta, delta.sum(0)]
        for i, grad in enumerate(grads):
            m[i] = .9*m[i] + .1*grad
            v[i] = .999*v[i] + .001*grad**2
            params[i] -= .003*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
    return params, len(y)


def replay(ep, policy, k, params):
    seen = set()
    checks = 0
    for step, graph in enumerate(ep.draws):
        if graph is None:
            continue
        chosen = graph
        if graph in seen and k:
            edits = [e for e in ep.edges if graph ^ {e} not in seen]
            if policy == "random":
                random.Random(f"{SEED}/{ep.name}/{step}").shuffle(edits)
            elif policy == "mlp" and edits:
                w, b, out, bias = params
                x = np.asarray([ep.features(graph, e) for e in edits])
                scores = (np.tanh(x @ w + b) @ out + bias).ravel()
                edits = [edits[i] for i in np.argsort(-scores, kind="stable")]
            for edit in edits[:k]:
                checks += 1
                candidate = graph ^ {edit}
                if ep.verify(candidate):
                    chosen = candidate
                    break
        assert chosen in ep.truth
        seen.add(chosen)
    return dict(task=ep.name, policy=policy, k=k, checks=checks, distinct=len(seen),
                recovery=len(seen)/len(ep.truth), correct=sum(g is not None for g in ep.draws))



# Each row: question ID, intervention signatures, twelve hexadecimal edge masks.
# A dash is an unparsed answer; zero is the valid representation of an empty graph.
SAMPLES = {

4: """n2_129 B4,D6 410 410 410 410 800 c00 410 410 410 c10 410 410
n2_102 Ae,D6 410 811 411 410 411 811 411 411 811 11 411 11
n2_128 B4,D4 810 10 810 410 810 410 410 810 10 800 2 10
n2_097 Ae,C2 111 87 87 80 103 103 3 103 3 3 103 23
n2_020 A0,D6 c00 c00 c00 c00 c00 c00 c00 c00 c00 c00 c00 c00
n2_161 B5,D5 a00 a18 a00 18 18 a18 18 418 a 418 a18 418
n2_132 B4,D7 840 40a 610 440 610 211 610 610 610 610 610 600
n2_202 Cb,D2 540 540 1c0 540 540 540 1c0 540 540 540 1c0 1c0
n2_096 Ae,C8 103 7 a2 103 103 103 182 103 103 103 7 103
n2_204 Cb,D3 7c0 1 7c0 3c0 1c0 1c0 1c0 3c0 2c0 1c0 700 1c0
n1_021 C1 40 40 40 40 40 40 40 40 40 40 40 40
n2_078 Aa,Ca 185 185 185 185 110 5 185 5 110 185 5 85
n2_151 B9,C1 68 68 68 68 40 68 28 28 28 28 68 68
n2_002 A0,B8 20 20 20 20 20 20 20 20 20 20 0 20
n2_137 B1,C0 8 100 8 8 8 8 8 8 8 0 8 8
n2_049 A4,D5 a00 202 202 202 a02 a02 202 202 202 202 202 202
n2_146 B1,D5 a08 a00 a08 808 808 8 a08 808 808 808 808 808
n2_075 Aa,C0 5 5 5 5 5 5 5 5 5 5 5 5
n1_001 A0 0 1 100 2 8 10 0 40 1 200 2 0
n2_155 B9,D1 228 208 28 208 28 208 828 28 28 208 228 220
n2_082 A6,B0 11 3 11 0 3 1 11 2 21 2 2 3
n1_002 A8 4 20 100 200 4 21 110 100 102 20 4 4
n1_013 B1 8 8 8 8 8 8 8 8 8 8 8 0
n2_022 A0,D5 a00 a00 a00 a00 a00 a00 a00 a00 a00 a00 a00 a00
n2_009 A0,C0 0 0 101 0 10 0 10 0 4 1 1 40
n2_053 Ac,B4 102 16 110 6 6 10 6 16 6 6 6 10
n2_016 A0,Cb 1c0 1c0 1c0 1c0 1c0 1c0 1c0 1c0 1c0 1c0 1c0 1c0
n2_188 C1,D0 40 840 40 40 40 40 0 0 40 40 40 40
n2_147 B1,D3 608 600 408 408 608 408 408 408 408 608 600 208
n2_120 B8,C8 100 100 120 20 101 100 121 120 120 120 a0 120
n2_043 A4,Bd 38 3a 18 118 1a 18 118 18 118 18 3a 18
n1_012 Bc 110 110 110 110 110 110 110 110 110 110 110 110
n2_098 Ae,Ca 187 87 107 7 5 7 180 103 103 5 103 103
n2_166 Bd,C9 178 118 78 150 118 118 118 118 118 118 118 118
n2_193 C1,D7 e00 840 e40 e00 840 840 a40 c40 c40 640 640 840
n2_099 Ae,D0 111 7 11 11 11 103 111 3 111 11 111 23
n2_031 A8,C0 4 104 0 4 100 4 4 4 100 4 4 4
n2_143 B1,D0 8 - 800 8 8 8 8 - 0 8 0 0
n2_114 B0,D6 410 880 880 410 800 410 410 c00 410 c00 c00 410
n2_089 A6,D6 11 410 410 410 410 410 410 410 410 410 410 410
n2_163 Bd,C0 38 18 28 118 18 38 118 8 18 18 28 118
n2_023 A0,D3 201 201 600 201 600 201 600 400 201 200 201 201
n1_027 D2 400 400 400 400 400 400 400 400 400 400 400 400
n2_121 B8,Ca 1a0 1a0 1a1 1a0 1a0 1a0 1a0 1a0 1a0 1a0 a0 1a0
n2_052 Ac,B8 26 820 26 820 26 6 20 26 26 26 26 26
n1_018 C8 100 100 100 100 100 100 100 100 100 100 100 100
n2_042 A4,B5 a a a a a a a a a a a a
n2_117 B0,D3 600 600 600 600 600 600 600 600 0 600 201 201
n2_185 C2,D7 e80 a80 680 a80 600 600 e00 400 610 e00 - 610
n2_181 C2,D2 480 0 480 480 480 480 480 480 480 480 480 480
n2_058 Ac,D0 c0 102 110 102 102 2 6 102 2 6 402 2
n2_159 B5,D4 818 818 818 818 818 18 800 18 18 818 818 18
n2_033 A8,C2 84 84 84 84 84 84 84 84 84 84 84 84
n2_162 B5,D7 610 818 218 418 18 418 418 418 618 418 618 418
n2_091 Ae,B0 7 1 7 11 111 111 3 7 23 11 4 111
n2_103 B0,C0 0 101 1 10 400 0 2 1 10 1 0 100
n2_012 A0,Ca 180 0 180 180 180 180 180 180 180 180 180 180
n2_148 B1,D7 e00 e08 e08 e08 208 e08 208 a08 e08 808 608 c08
n2_036 A8,Cb 1c4 c4 c4 c4 c4 c4 c4 c4 c4 0 c4 c4
n2_013 A0,C1 40 40 40 40 40 40 40 40 0 40 40 40
n2_180 C2,D0 80 80 80 80 80 80 10 80 80 80 80 80
n2_055 Ac,Bd 3e 11a e e e 10a e 10a a a a 10a
n2_108 B0,C9 140 0 140 140 140 140 140 140 140 140 140 140
n2_144 B1,D4 808 808 808 808 808 808 808 808 808 808 808 808
n1_010 B8 20 20 20 20 20 20 20 20 20 20 20 20
n2_021 A0,D1 200 200 200 200 200 200 201 202 0 200 200 200
n2_192 C1,D3 640 640 0 640 640 640 640 640 640 640 640 640
n2_035 A8,C9 44 121 300 44 44 44 44 44 44 44 44 44
n2_176 C0,D5 a00 a00 a00 a00 0 a00 a00 a00 a00 a00 0 a00
n2_047 A4,D2 410 410 410 410 410 410 410 402 402 410 402 402
n2_057 Ac,C8 100 102 6 102 102 102 102 102 102 102 102 102
n1_023 C3 c0 c0 c0 c0 c0 c0 c0 c0 c0 c0 c0 c0
n2_068 A2,D4 800 801 801 801 801 801 801 801 801 801 801 801
n2_044 A4,C0 2 100 2 2 2 2 2 0 2 2 2 2
n2_125 B8,D0 20 20 20 20 20 20 110 100 20 100 4 20
n2_113 B0,D2 400 1 400 400 400 400 400 400 400 400 400 400
n2_019 A0,D2 400 400 400 400 400 880 400 400 400 400 400 0
n2_088 A6,D2 403 412 403 403 403 403 403 0 403 403 403 403
n2_060 A2,B0 1 0 1 - 1 1 1 0 1 1 8 1
n1_025 D0 0 1 2 10 100 0 1 2 4 1 0 10
n2_045 A4,D0 2 202 2 2 10 2 800 2 10 2 - -
n2_150 B9,C8 128 108 108 100 128 128 128 28 128 100 108 108
n2_064 A2,Ca 181 81 181 81 181 181 181 181 181 181 181 181
n2_018 A0,D4 800 880 800 0 800 800 800 800 800 800 800 800
n1_030 D5 a00 a00 a00 a00 a00 a00 a00 a00 a00 a00 a00 a00
n2_171 C0,D0 0 0 0 0 1 0 0 1 2 8 100 0
n2_177 C0,D3 600 600 600 600 600 600 600 600 600 200 200 200
n2_186 Ca,D0 180 180 180 180 180 180 180 180 180 180 180 180
n2_142 B1,Cb 1c0 1c8 a c8 8 1c8 a 88 1c8 102 1c0 148
n2_100 Ae,D4 111 880 880 880 111 880 880 880 880 880 880 880
n2_165 Bd,C1 50 38 118 38 18 18 18 18 18 118 48 18
n2_122 B8,C1 60 60 60 60 60 60 60 60 60 60 60 60
n2_063 A2,C2 81 81 81 81 81 81 81 0 81 81 81 81
n2_167 Bd,D0 38 18 18 38 18 118 18 18 118 18 118 118
n2_170 Bd,D5 238 238 38 218 218 38 218 818 818 218 18 18
n2_093 Ae,B4 11 17 111 15 7 11 11 11 11 11 15 15
n2_029 A8,B9 2c c c c c 2c c c c c c c
n2_133 Bc,C0 110 110 110 110 110 0 110 110 110 110 110 110
n1_029 D1 200 200 200 200 200 200 200 200 200 200 200 200
n2_141 B1,C3 c0 c0 c0 c0 c0 c0 c0 c0 c0 c0 c0 c0
n2_182 C2,D6 c80 0 880 c00 c00 c80 480 480 c00 c80 880 480
n2_106 B0,Ca 180 18 180 180 180 180 180 180 180 180 0 180
n2_194 C9,D0 140 140 140 140 140 140 140 140 140 140 140 140
n2_158 B5,D0 18 18 600 18 18 18 18 18 18 18 18 18
n2_130 B4,D1 210 210 210 210 0 210 210 210 210 0 210 210
n2_062 A2,C8 101 101 101 101 101 101 101 101 101 101 101 101
n2_101 Ae,D2 410 450 450 410 400 410 410 410 418 410 410 410
n2_065 A2,C3 c0 c0 41 41 41 41 41 41 41 41 41 41
n2_135 Bc,D0 - 110 10 10 110 10 110 110 110 10 110 30
n2_069 A2,D2 401 101 400 400 400 401 - 1 401 - 0 10
n2_085 A6,C2 80 88 80 82 80 3 80 82 82 82 82 83
n2_048 A4,D6 c00 c00 c02 410 c00 402 402 410 c00 410 410 410
n2_072 A2,D7 a01 e00 201 201 201 a01 201 a01 a01 a01 a01 a01
n2_010 A0,C8 100 100 100 100 100 100 100 100 100 100 100 0
n2_059 Ac,D4 102 840 800 800 840 800 800 102 800 800 802 6
n2_083 A6,B4 11 10 11 3 10 11 0 23 11 10 11 11
n2_017 A0,D0 0 1 2 4 200 0 1 2 8 10 800 0
n2_094 Ae,Bc 110 111 82 11 11 111 11 111 11 11 111 102
n2_153 B9,Cb e0 1c0 a8 a8 e0 a8 68 e8 a8 68 a8 a8
n2_056 Ac,C0 102 102 6 102 102 0 6 102 6 2 2 102
n2_115 B0,D1 200 200 200 - 200 200 200 200 200 200 200 200
n2_109 B0,C3 c0 c0 c0 40 c0 40 c0 c0 c0 c0 c0 0
n2_038 A4,B0 2 2 0 2 2 2 2 2 2 2 2 2
n2_112 B0,D4 800 800 0 800 0 800 800 800 800 - 800 800
n2_203 Cb,D1 3c0 3c0 1c0 1c0 1c0 1c0 1c0 3c0 3c0 3c0 3c0 1c0
n2_092 Ae,B8 111 21 21 111 23 23 a2 21 23 11 111 23
n2_195 C9,D1 240 340 240 140 140 140 140 340 340 140 140 140
n2_084 A6,C0 3 1 3 3 3 1 3 3 3 3 3 3
n2_172 C0,D4 800 800 - 0 800 800 800 800 - 800 800 40
n2_123 B8,C9 60 160 60 160 160 140 160 140 160 140 160 160
n2_160 B5,D1 202 210 218 208 18 18 218 218 218 208 218 18
n2_139 B1,C1 40 48 48 48 40 48 40 48 48 48 48 48
n2_025 A8,B0 100 4 20 2 4 100 10 4 4 20 4 0
n2_030 A8,Bd 3c 118 102 1c 10a 10a 118 60 1c 1c 118 1c
n2_152 B9,C9 168 - 10a 10a a 108 10a 168 a 168 168 168
n2_076 Aa,C8 105 105 0 105 105 105 105 105 105 105 105 105
n2_104 B0,C8 100 100 100 100 100 0 0 100 100 100 100 100
n2_126 B4,C0 100 10 10 10 10 10 10 10 50 10 10 10
n2_074 Aa,B8 25 5 24 5 5 21 14 30 20 110 20 20
n2_190 C1,D1 240 240 240 240 0 240 240 240 240 240 240 240
n2_134 Bc,C8 110 110 110 110 0 110 110 110 110 110 110 110
n1_011 B4 10 10 10 10 10 10 10 10 10 10 10 10
n1_017 C0 0 1 10 1 40 0 4 200 2 8 0 80
n2_004 A0,Bc 110 110 110 110 10 110 10 110 10 110 110 10
n2_149 B9,C0 28 28 28 28 28 28 28 28 28 28 28 28
n2_051 Ac,B0 102 10 102 102 0 102 102 2 102 102 0 6
n2_087 A6,D4 810 11 811 801 0 800 11 402 11 801 800 802
n1_004 Ac 102 102 102 102 102 102 102 102 102 102 102 102
n1_008 Ae 7 31 11 111 121 11 31 23 82 111 82 31
n2_107 B0,C1 40 40 40 40 40 40 40 40 40 41 0 40
n2_040 A4,B4 12 12 12 0 0 12 12 12 12 12 12 12
n2_105 B0,C2 80 80 80 80 80 80 80 80 0 80 80 80
n2_077 Aa,C2 85 85 85 5 5 5 85 85 85 85 85 85
n2_175 C0,D1 200 200 200 200 200 200 200 200 200 200 200 210
n2_086 A6,D0 11 200 3 11 2 3 1 2 2 0 3 11
n1_020 Ca a0 a0 a0 a0 a0 a0 a0 a0 a0 a0 a0 a0
n2_003 A0,B4 10 10 10 10 10 10 10 0 10 10 10 10
n1_022 C9 140 140 140 140 140 140 140 140 140 140 140 140
n2_131 B4,D5 a10 a10 a10 a10 a10 a10 a10 210 a10 210 210 210
n2_027 A8,B4 14 14 14 10 14 14 14 14 14 14 0 102
n2_070 A2,D6 c01 c01 c01 c01 c01 c01 c01 c00 c00 c01 400 400
n2_095 Ae,C0 7 7 7 103 103 3 3 7 3 7 3 7
n1_006 Aa 21 5 11 11 a2 21 21 21 5 111 21 5
n2_073 Aa,B0 21 5 21 5 5 5 11 5 5 5 21 5
n1_024 Cb 1c0 141 c4 45 45 60 45 45 45 61 45 45
n1_028 D6 410 211 410 410 410 410 410 410 410 410 410 410
n2_006 A0,B9 28 28 28 28 28 28 28 28 28 0 28 28
n2_071 A2,D3 600 201 - 201 201 201 201 201 201 201 201 201
n2_046 A4,D4 800 800 802 800 802 800 802 0 2 802 2 800
n1_015 B5 18 18 18 18 18 18 18 18 18 18 18 18
n2_080 Aa,D0 5 5 5 7 5 5 5 5 5 5 5 5
n2_015 A0,C3 c0 c0 c0 c0 c0 c0 c0 c0 c0 c0 c0 c0
n2_189 C1,D2 400 440 440 440 440 440 440 440 440 440 440 440
n2_124 B8,Cb 141 1c0 1c0 141 140 c0 c0 c0 e0 e0 e0 e0
n1_007 A6 3 11 1 3 3 11 11 3 3 11 11 3
n2_173 C0,D2 400 400 400 0 400 400 400 400 400 400 400 400
n2_026 A8,B8 24 14 100 4 14 24 24 14 20 14 20 24
n2_178 C0,D7 e00 e00 e00 e00 e00 e00 e00 e00 0 e00 e00 e00
n2_037 A8,D0 100 4 800 10 0 102 4 200 800 100 0 4
n1_014 B9 28 28 28 28 28 28 28 28 28 28 28 28
n2_014 A0,C9 140 140 140 140 140 140 140 140 140 140 140 140
n2_174 C0,D6 410 c00 880 410 400 c10 410 410 410 400 410 410
n2_005 A0,B1 8 8 8 8 8 8 8 8 8 8 8 8
n1_026 D4 800 c0 800 840 840 800 800 800 800 800 800 800
n1_032 D7 e00 211 211 211 841 211 211 203 203 203 211 203
n2_184 C2,D3 600 a80 400 a80 680 680 600 600 600 600 600 680
n2_140 B1,C9 140 140 140 111 148 148 148 148 148 148 148 148
n2_111 B0,D0 0 1 2 4 8 0 40 1 400 80 0 2
n2_001 A0,B0 0 0 0 1 10 2 100 0 1 10 2 4
n2_090 A6,D7 211 211 211 211 211 211 211 211 211 211 211 11
n2_191 C1,D5 a00 840 840 840 a40 a40 a40 a40 a40 a40 a40 a40
n2_061 A2,C0 1 1 1 1 1 81 1 1 1 1 1 41
n2_179 C8,D0 100 0 - 100 100 100 100 100 100 100 100 0
n1_019 C2 80 80 80 80 80 80 80 80 80 80 80 80
n2_024 A0,D7 e00 602 8c0 201 200 200 201 201 211 211 211 201
n2_157 B5,C1 50 18 10 18 18 48 18 58 48 40 48 18
n2_196 C3,D0 c0 c0 c0 c0 c0 c0 c0 c0 c0 c0 c0 c0
n2_079 Aa,Cb 1c0 45 1c0 1c0 141 45 45 45 45 45 45 45
n1_009 B0 0 1 1 40 1 0 10 2 40 1 1 101
n2_119 B8,C0 100 120 20 20 - 0 20 20 20 20 a0 20
n2_169 Bd,D1 238 238 38 238 38 38 38 18 3a 38 18 38
n1_031 D3 600 600 600 600 600 600 600 600 600 600 600 600
n1_016 Bd 118 118 38 150 118 108 18 118 18 118 118 118
n2_118 B0,D7 e00 200 201 600 201 600 602 202 e00 600 600 211
n2_127 B4,D0 10 410 10 10 10 10 10 10 10 10 10 10
n2_197 C3,D2 4c0 0 4c0 4c0 4c0 c0 0 4c0 4c0 c0 c0 4c0
n2_136 Bc,D4 - 820 800 810 800 800 810 830 810 820 810 810
n2_034 A8,Ca 184 184 184 184 184 184 184 184 84 184 184 184
n2_201 Cb,D0 1c0 c0 c0 1c0 1c0 1c0 1c0 c0 1c0 1c0 1c0 1c0
n2_050 A4,D7 22 222 202 202 211 880 202 880 202 e00 202 282
n2_154 B9,D0 220 28 20 20 20 28 28 28 28 28 28 28
n2_007 A0,B5 18 0 18 18 18 18 18 18 18 18 18 18
n2_008 A0,Bd 38 18 28 18 18 118 38 118 18 18 18 18
n2_081 Aa,D2 405 401 5 401 5 5 404 404 5 5 404 21
n1_005 A2 1 1 1 1 1 1 1 1 1 1 1 1
n2_183 C2,D1 280 280 280 280 280 280 280 280 280 280 0 280
n2_138 B1,C8 100 100 108 108 108 108 108 108 108 108 108 108
n2_032 A8,C8 104 104 100 100 100 0 104 100 14 104 100 14
n2_168 Bd,D4 38 800 38 838 838 38 838 38 38 838 38 38
n1_003 A4 2 2 10 2 10 2 2 2 100 11 2 2
n2_198 C3,D1 2c0 2c0 2c0 2c0 2c0 2c0 2c0 2c0 2c0 2c0 2c0 2c0
n2_164 Bd,C8 138 100 118 118 0 118 118 118 138 138 118 118
n2_156 B5,C0 18 18 18 18 18 18 18 18 18 18 18 18
n2_187 Ca,D2 580 180 580 180 0 480 180 400 480 180 180 480
n2_200 C3,D7 11 8c0 2c0 6c0 8c0 8c0 8c0 8c0 2c0 ac0 e00 4c0
n2_039 A4,B8 20 22 22 22 22 22 22 22 22 22 22 22
n2_110 B0,Cb 1c0 1c0 1c0 1c0 1c0 1c0 1c0 0 1c0 1c0 1c0 1c0
n2_041 A4,Bc 30 12 32 10 30 30 12 32 12 10 32 12
n2_145 B1,D1 208 208 208 208 0 208 208 208 208 208 208 0
n2_116 B0,D5 a00 a00 a00 a00 a00 a00 a00 a00 a00 a00 a00 a00
n2_067 A2,D0 1 1 1 1 1 1 0 0 1 1 1 1
n2_199 C3,D3 1 6c0 c0 440 2c0 8c0 2c0 8c0 2c0 4c0 8c0 2c0
n2_011 A0,C2 80 80 80 80 80 80 80 80 80 80 80 80
n2_028 A8,Bc 110 110 114 14 114 10 14 110 110 14 114 14
n2_054 Ac,Bc 12 10a 112 112 112 112 112 112 110 112 110 110
n2_066 A2,Cb 1c0 c1 141 141 141 141 141 141 141 141 188 141""",

5: """five_23 D5,Ef 4041 42 30002 f5000 f0002 35000 85000 15000 b5000 15000 25000 15000
five_28 A18,Bc 42c 2c 2c c 2c 2c 2c c 2c 2c 2c 2c
five_01 B1,C9 500 510 510 510 10 510 510 510 510 510 510 510
five_03 Aa,B0 5 5 5 5 0 5 5 5 5 5 5 5
five_06 B8,Ec 40440 40440 40440 40040 40040 40440 40040 40040 40440 40040 40440 40440
five_20 D5,E4 45000 45000 45000 45000 45000 45000 45000 45000 45000 5000 5000 45000
five_14 C10,D11 9800 9800 9800 9800 1800 9000 1800 9800 1800 9000 1800 1800
five_15 A12,C0 9 9 9 9 9 9 9 9 9 9 9 9
five_04 B5,E4 40030 8422 40030 40030 40030 40030 40030 40030 30 40030 30 30
five_19 B0,E6 20020 20020 20020 20020 0 20020 20020 20020 20020 20020 20020 20020
five_26 B8,E0 40 40 40 40 40 40 40 40 40 20040 40 40
five_11 A0,C8 400 400 400 0 400 400 400 400 400 400 400 400
five_13 A0,D2 2000 2000 2000 2000 2000 2000 2000 2000 2000 2000 2000 2000
five_02 A1e,D0 f b b b b f 7 7 3 3 87 3
five_16 A2,E2 20001 1 10001 20000 1 20000 20001 20001 20001 20001 20001 20001
five_21 A8,C12 a04 a04 a04 a04 a04 a04 a04 a04 0 0 a04 a04
five_27 Ae,D0 7 7 7 7 3 7 7 3 43 7 3 43
five_00 B5,D14 c030 c000 8030 4030 8030 30 c030 30 c030 c030 c030 c030
five_24 A10,B10 88 48 28 88 28 48 28 28 88 88 48 28
five_25 C0,E5 50000 50000 50000 50000 50000 50000 50000 50000 50000 50000 50000 50000
five_08 B8,C8 440 440 440 440 440 440 440 440 440 440 440 440
five_17 Bc,Ec 40420 40420 40420 40420 40420 20420 80420 420 420 40420 420 40420
five_09 B5,D7 3020 3030 3 2030 2030 2030 2030 2012 1030 3020 1030 3020
five_22 Ac,E8 80006 80002 80002 80006 80002 80006 80002 80006 80006 80002 80006 80006
five_18 Aa,C10 805 805 805 805 805 805 805 805 805 805 805 805
five_29 B9,C18 8450 c50 c50 8450 8450 c50 450 450 c50 c50 c50 450
five_05 D1,E9 81000 81000 91000 81000 81000 91000 91000 81000 81000 11000 81000 81000
five_07 D4,Ec 84000 44100 84100 84000 44000 44100 84100 84000 44000 44000 44000 84000
five_10 D1,E2 21000 21000 21000 21000 21000 21000 21000 21000 21000 21000 21000 21000
five_12 B0,D12 a000 a000 a000 a000 a000 a000 a000 a000 a000 a000 a000 a000""",

}


def train_four(groups, output):
    # Balance fold sizes without using any labels or performance data.
    folds = [[] for _ in range(5)]
    for family in sorted(groups, key=lambda f: (-len(groups[f]), f)):
        min(folds, key=len).extend(groups[family])
    results, splits = [], []
    for i, test in enumerate(folds):
        train = [ep for j, fold in enumerate(folds) if i != j for ep in fold]
        assert not {e.family for e in train} & {e.family for e in test}
        params, n = fit(train)
        splits.append(dict(fold=i, train_examples=n, test_tasks=[e.name for e in test]))
        for ep in test:
            for policy, k in [("original", 0)]+[(p,k) for k in (1,3) for p in ("fixed","random","mlp")]:
                results.append(dict(fold=i, **replay(ep, policy, k, params)))
    summary = []
    for policy, k in sorted({(r["policy"],r["k"]) for r in results}):
        rows = [r for r in results if (r["policy"],r["k"]) == (policy,k)]
        assert len(rows) == sum(map(len, folds))
        summary.append(dict(policy=policy,k=k,recovery=np.mean([r["recovery"] for r in rows]),
                            **{f:sum(r[f] for r in rows) for f in ("distinct","checks","correct")}))
    output.mkdir(exist_ok=True)
    params, n = fit(sum(folds, []))
    np.savez(output/"edit-mlp.npz", w=params[0], b=params[1], out=params[2], bias=params[3])
    report = dict(families=len(groups), full_fit_examples=n, summary=summary, splits=splits, episodes=results)
    (output/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:report[k] for k in ("families","full_fit_examples","summary")},indent=2))



def evaluate(output):
    train_four(load_groups(4, output), output)
    weights = output/"edit-mlp.npz"
    with np.load(weights) as saved:
        params = [saved[k] for k in ("w", "b", "out", "bias")]
    groups = load_groups(5, output)
    episodes = [ep for family in groups.values() for ep in family]
    rows, summary = [], []
    settings = [("original", 0)] + [(p,k) for k in (1,3) for p in ("fixed","random","mlp")]
    for policy,k in settings:
        batch = [replay(ep,policy,k,params) for ep in episodes]
        rows.extend(batch)
        summary.append(dict(policy=policy,k=k,recovery=np.mean([r["recovery"] for r in batch]),
                            **{f:sum(r[f] for r in batch) for f in ("correct","distinct","checks")}))
    report = dict(train_nodes=4, test_nodes=5, tasks=len(episodes), families=len(groups),
                  weights_sha256=sha256(weights.read_bytes()).hexdigest(), summary=summary, episodes=rows)
    (output/"five-node-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary,indent=2))


def questions(size):
    nodes = tuple("ABCDE"[:size])
    for line in SAMPLES[size].splitlines():
        name, signature, *draws = line.split()
        observations = []
        for encoded in signature.split(","):
            mask = int(encoded[1:], 16)
            effects = {node: (mask >> i) & 1 for i, node in enumerate(nodes)}
            observations.append(dict(perturbed_node=encoded[0], effects=effects))
        yield name, observations, [None if d == "-" else int(d, 16) for d in draws]


def dag_archive(nodes, edges):
    """Enumerate DAGs once across all topological orders, including isolated nodes."""
    bits = {e: 1 << i for i, e in enumerate(edges)}
    masks = set()
    for order in permutations(nodes):
        subsets = [0]
        for edge in combinations(order, 2):
            subsets += [m | bits[edge] for m in subsets]
        masks.update(subsets)
    for mask in sorted(masks):
        arcs = [e for i,e in enumerate(edges) if mask & (1 << i)]
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(arcs)
        yield mask, arcs, {n:nx.descendants(g,n) for n in nodes}


def load_groups(size, output):
    nodes = tuple("ABCDE"[:size])
    edges = tuple(permutations(nodes, 2))
    archive = list(dag_archive(nodes, edges))
    if len(archive) != {4:543, 5:29281}[size]:
        raise ValueError("Incomplete DAG archive")
    checkpoint = output/f"draws-{size}.json"
    fresh = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    groups = defaultdict(list)
    for name, observations, recorded in questions(size):
        required = [(o["perturbed_node"], {n for n,v in o["effects"].items() if v})
                    for o in observations]
        truth = {mask:{"edges":arcs} for mask, arcs, reach in archive
                 if all(reach[n] == targets for n, targets in required)}
        draws = recorded if fresh is None else fresh.get(name, [])
        if len(draws) != 12:
            raise ValueError(f"Incomplete baseline for {name}. Resume with --collect {size}.")
        rows = [dict(edges=None if m is None else [e for i,e in enumerate(edges) if m & (1<<i)],
                     valid=m in truth) for m in draws]
        item = dict(observation_set_id=name, observations=observations,
                    ground_truth_graphs=truth.values())
        ep = TransferEpisode(item, rows, nodes)
        groups[ep.family].append(ep)
    return groups


def format_edges(mask, edges):
    return ", ".join(f"{s}->{t}" for i,(s,t) in enumerate(edges) if mask & (1<<i)) or "No edges"


def parse_mask(text, edges):
    match = re.search(r"(?im)^\s*graph:\s*(.*)$", text)
    lines = text.strip().splitlines()
    answer = (match[1] if match else lines[0] if lines else "").strip().rstrip(".;").replace("→", "->")
    if answer.casefold() == "no edges":
        return 0
    selected = set()
    for part in answer.split(","):
        edge = tuple(n.strip() for n in part.split("->"))
        if edge not in edges:
            return None
        selected.add(edge)
    if nx.is_directed_acyclic_graph(nx.DiGraph(list(selected))):
        return sum(1 << edges.index(e) for e in selected)
    return None


def collect(size, output):
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("Set DASHSCOPE_API_KEY before collecting new answers.")
    nodes = tuple("ABCDE"[:size])
    edges = tuple(permutations(nodes, 2))
    path = output/f"draws-{size}.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    for index, (name, observations, _) in enumerate(questions(size)):
        draws = saved.setdefault(name, [])
        seeds = random.Random(SEED+index)
        for _ in draws:
            seeds.randrange(2**31)
        obs_text = "\n".join("Perturb({}) -> {}".format(o["perturbed_node"],
            " ".join(f"{n}:{v}" for n,v in o["effects"].items())) for o in observations)
        while len(draws) < 12:
            history = [format_edges(mask, edges) for mask in draws if mask is not None][-4:]
            prompt = (f"Find one directed acyclic graph on nodes {', '.join(nodes)}.\n"
                "After perturbing a node, all nodes reachable from it along arrows have value 1. "
                "Every other node, including the perturbed node, has value 0.\n"
                f"Observations:\n{obs_text}\nEarlier answers to avoid repeating:\n"
                + ("\n".join(history) or "None yet") + "\nUse only the given nodes. "
                "No cycles or self-loops. Output one line in the format Graph: X->Y, Y->Z or Graph: No edges.")
            system = ("Respond with exactly one plain-text line starting with Graph: followed by "
                      "the comma-separated directed edges or No edges. Omit all analysis and Markdown.")
            body = dict(model=MODEL, temperature=0.8, top_p=0.95, max_tokens=96,
                        enable_thinking=False, seed=seeds.randrange(2**31),
                        messages=[dict(role="system", content=system), dict(role="user", content=prompt)])
            request = Request(ENDPOINT, data=json.dumps(body).encode(),
                              headers={"Content-Type":"application/json"})
            request.add_unredirected_header("Authorization", "Bearer "+key)
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
            if result.get("model") != MODEL:
                raise RuntimeError("Unexpected model in API response")
            draws.append(parse_mask(result["choices"][0]["message"]["content"] or "", edges))
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(saved))
            temp.replace(path)
        print(f"{index+1}: {name} complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--collect", type=int, choices=(4,5), help="Run a new API baseline instead of offline replay.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(args.collect, args.output)
    else:
        evaluate(args.output)


