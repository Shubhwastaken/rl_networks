"""
Debug: Trace where LP-violating bounds originate.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from fixed_environment import PartitionBoundEnv, _compute_partition_bound, Phase
from fixed_graph_generation import get_all_graph_infos, identify_graph
from fixed_inequality import EntropyIndex, make_fractional
from fixed_base_inequality_generator import internal_per_partition
from lp_lower_bound import compute_lp_lower_bound

env = PartitionBoundEnv(graph_dataset_size=5, stage=4)

# Pick butterfly_8N
for graph_tuple in env.graph_dataset:
    nodes, edges, sessions = graph_tuple
    name = identify_graph(nodes, edges, sessions)
    if 'butterfly' in name or 'paper' in name:
        break

print(f"Graph: {name}")
print(f"PB = {_compute_partition_bound(nodes, edges, sessions):.4f}")
print(f"LP LB = {compute_lp_lower_bound(nodes, edges, sessions):.4f}")

# Simulate Phase 3 setup
from partition import generate_random_valid_partition, decode_partition
import networkx as nx
from collections import defaultdict

G = nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from(edges)
col = nx.coloring.greedy_color(G, strategy='largest_first')
groups = defaultdict(list)
for nd, c in col.items(): groups[c].append(nd)
partition = list(groups.values())

env.nodes = nodes
env.edges = edges
env.sessions = sessions
env.adjacency = {n: set() for n in nodes}
env.edge_set = set()
for u, v in edges:
    env.adjacency[u].add(v); env.adjacency[v].add(u)
    env.edge_set.add((u,v)); env.edge_set.add((v,u))
env.partition = partition
env.partition_weights = {}
env.assignment = {}
env.num_groups = len(partition)
env._assignment_complete = True
env._refinement_steps = 0
env.prev_internal_count = 0
env.partition_bound = _compute_partition_bound(nodes, edges, sessions)

env._start_phase2()
env._start_phase3()
env.internal_per_part = env.internal_per_part or []

ipp = internal_per_partition(partition, sessions)
print(f"\nPartition: {partition}")
print(f"Internal per part: {ipp}")

# Check what's in frac_pool immediately after init
print(f"\n=== frac_pool after _start_phase3 ({len(env.frac_pool)} items) ===")
for i, ineq in enumerate(env.frac_pool):
    is_term = ineq.check_valid_terminal_form()
    if is_term:
        b = ineq.extract_bound(len(sessions), len(edges), ipp)
        print(f"  [{i}] TERMINAL bound={b:.4f}: {str(ineq)[:120]}")
    else:
        # Check WHY it's not terminal
        yi = ineq.get_yi_coefficient()
        has_yst = bool(ineq.active_yst())
        rhs_edge = ineq.get_rhs_edge_coefficient()
        has_pos_src = any(ineq.coeffs[ineq.index.get_source_idx(v)] > 1e-4 for v in ineq.index.nodes)
        has_neg_src = any(ineq.coeffs[ineq.index.get_source_idx(v)] < -1e-4 for v in ineq.index.nodes)
        print(f"  [{i}] non-terminal: yi={yi:.3f} yst={has_yst} edge={rhs_edge:.3f} +src={has_pos_src} -src={has_neg_src}")

best = env.frac_pool.best_bound(len(sessions), len(edges), ipp)
print(f"\nbest_bound = {best:.4f} (PB = {env.partition_bound:.4f})")
print(f"LP LB = {compute_lp_lower_bound(nodes, edges, sessions):.4f}")

if best < env.partition_bound - 1e-8:
    print(f"\n!! IMMEDIATE sub-PB bound from frac_pool init!")
    for ineq in env.frac_pool:
        if ineq.check_valid_terminal_form():
            b = ineq.extract_bound(len(sessions), len(edges), ipp)
            if abs(b - best) < 1e-9:
                print(f"   Offending inequality: {ineq}")
                print(f"   Coefficients: {ineq.coeffs}")
                break

# Also check Stage 3 abs(total_reward) issue
print(f"\n=== Stage 3 fallback check ===")
print(f"_best_pool_bound after Phase 2 setup:")
bp = env._best_pool_bound()
print(f"  _best_pool_bound = {bp}")
print(f"  Would fallback to abs(total_reward) if None: {'YES - THIS IS THE BUG' if bp is None else 'no'}")
