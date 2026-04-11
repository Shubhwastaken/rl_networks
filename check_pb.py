from fixed_graph_generation import _build_registry, GRAPH_REGISTRY
from fixed_environment import _compute_partition_bound

_build_registry()
for info in GRAPH_REGISTRY:
    env_pb = _compute_partition_bound(info.nodes, info.edges, info.sessions)
    opt = info.optimal_bound
    tag = "MISMATCH" if abs(opt - env_pb) > 1e-6 else "OK"
    print(f"  {info.name:20s} opt={opt:.4f}  env_PB={env_pb:.4f}  {tag}")
