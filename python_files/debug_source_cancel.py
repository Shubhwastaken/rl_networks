"""
Debug script: check what cancel_source_terms produces vs raw extract_bound.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from fixed_graph_generation import get_all_graph_infos
from fixed_inequality import EntropyIndex, Inequality
from fixed_base_inequality_generator import (
    generate_base_inequalities, generate_all_node_ios,
    internal_per_partition
)
from fixed_submodularity import apply_n2_submodularity_all_at_once, apply_pairwise_submodularity
from fixed_environment import _compute_partition_bound

infos = get_all_graph_infos()
# Pick a simple graph
for info in infos:
    if info.name == 'paper_7N':
        nodes, edges, sessions = list(info.nodes), list(info.edges), list(info.sessions)
        break

print(f"Graph: paper_7N, {len(nodes)} nodes, {len(edges)} edges, {len(sessions)} sessions")
print(f"Sessions: {sessions}")
print(f"PB = {_compute_partition_bound(nodes, edges, sessions):.4f}")

# Make a simple 2-partition
from partition import generate_random_valid_partition, decode_partition
import networkx as nx
from collections import defaultdict

G = nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from(edges)
col = nx.coloring.greedy_color(G, strategy='largest_first')
groups = defaultdict(list)
for nd, c in col.items(): groups[c].append(nd)
partition = list(groups.values())

print(f"\nPartition: {partition}")
index = EntropyIndex(partitions=partition, nodes=nodes, edges=edges, sessions=sessions)
print(f"Index dim: {index.dim}")
print(f"Variables:")
for idx, name in sorted(index.idx_to_var.items()):
    print(f"  [{idx}] {name}")

# Generate base inequalities
base_ineqs = generate_base_inequalities(partition, nodes, edges, sessions, index)
print(f"\n--- Base (partition-level) inequalities ---")
for i, ineq in enumerate(base_ineqs):
    print(f"  IO(P{i}): {ineq}")

# Generate node IOs
node_ios = generate_all_node_ios(partition, nodes, edges, sessions, index)
print(f"\n--- Node IOs ---")
for node, fi in sorted(node_ios.items()):
    print(f"  IO({node}): {fi}")
    print(f"    Y_I coeff = {fi.coeffs[index.yi_idx()]:.4f}")
    print(f"    check_valid_terminal = {fi.check_valid_terminal_form()}")
    # Try extract_bound ignoring terminal check
    ipp = internal_per_partition(partition, sessions)
    b = fi.extract_bound(len(sessions), len(edges), ipp)
    print(f"    extract_bound (raw) = {b:.4f}")
    cancelled = fi.cancel_source_terms()
    print(f"    After cancel_source_terms: {cancelled}")
    print(f"    check_valid_terminal (cancelled) = {cancelled.check_valid_terminal_form()}")
    if cancelled.check_valid_terminal_form():
        b2 = cancelled.extract_bound(len(sessions), len(edges), ipp)
        print(f"    extract_bound (cancelled) = {b2:.4f}")

# N2 submod result
print(f"\n--- N2 submod result ---")
n2 = apply_n2_submodularity_all_at_once(base_ineqs, index, sessions)
print(f"  {n2}")
print(f"  check_valid_terminal = {n2.check_valid_terminal_form()}")
ipp = internal_per_partition(partition, sessions)
b = n2.extract_bound(len(sessions), len(edges), ipp)
print(f"  extract_bound = {b:.4f}")

# Now sum ALL node IOs and try cancel
print(f"\n--- Sum of all node IOs ---")
combined = list(node_ios.values())[0].copy()
for fi in list(node_ios.values())[1:]:
    combined = combined.add(fi)
print(f"  Sum: {combined}")
print(f"  Y_I coeff = {combined.coeffs[index.yi_idx()]:.4f}")
print(f"  check_valid_terminal = {combined.check_valid_terminal_form()}")

cancelled = combined.cancel_source_terms()
print(f"  After cancel: {cancelled}")
print(f"  Y_I coeff (cancelled) = {cancelled.coeffs[index.yi_idx()]:.4f}")
print(f"  check_valid_terminal (cancelled) = {cancelled.check_valid_terminal_form()}")
if cancelled.check_valid_terminal_form():
    b = cancelled.extract_bound(len(sessions), len(edges), ipp)
    print(f"  extract_bound (cancelled) = {b:.4f}")

# Try pairwise submod on two node IOs
print(f"\n--- Pairwise submod on node IOs ---")
node_list = sorted(node_ios.keys())
for i in range(len(node_list)):
    for j in range(i+1, len(node_list)):
        u, v = node_list[i], node_list[j]
        a = node_ios[u]
        b = node_ios[v]
        union_ineq = apply_pairwise_submodularity(a, b, index, sessions)
        if union_ineq.check_valid_terminal_form():
            bound = union_ineq.extract_bound(len(sessions), len(edges), ipp)
            print(f"  Union({u},{v}): terminal=True, bound={bound:.4f}")
            print(f"    {union_ineq}")
