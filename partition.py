import random


def generate_random_valid_partition(nodes, edges):
    """
    Generates one random valid partition as a chromosome.
    Each node gets a color (integer) such that no two adjacent
    nodes share the same color — enforcing the independent set
    constraint required by the partition bound proof.

    FIX: Added random.shuffle on nodes before coloring to break
         the fixed-order greedy bias. Previously nodes were always
         processed in their original list order, producing a biased
         distribution of partitions. Shuffling gives more diverse
         partitions across training episodes.

    Args:
        nodes : list of node names
        edges : list of edges as sorted tuples (u, v)

    Returns:
        chromosome: list of integer color ids, one per node,
                    in the same order as nodes
    """

    # build adjacency list
    adjacency = {n: set() for n in nodes}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    # shuffle node order to remove greedy coloring bias
    shuffled_nodes = nodes[:]
    random.shuffle(shuffled_nodes)

    colors = {}
    for node in shuffled_nodes:
        neighbor_colors = {
            colors[n]
            for n in adjacency[node]
            if n in colors
        }
        used_colors = set(colors.values())

        # always allow one new color beyond current max
        new_color = max(used_colors) + 1 if used_colors else 0
        candidate_colors = list(used_colors | {new_color})
        valid_colors = [
            c for c in candidate_colors
            if c not in neighbor_colors
        ]

        # valid_colors always contains new_color at minimum
        # so it is never empty
        chosen_color = random.choice(valid_colors)
        colors[node] = chosen_color

    # return chromosome in original node order (not shuffled order)
    # so chromosome[i] always corresponds to nodes[i]
    chromosome = [colors[n] for n in nodes]
    return chromosome


def decode_partition(nodes, chromosome):
    """
    Converts a chromosome into a list of groups.

    Args:
        nodes     : list of node names
        chromosome: list of integer color ids (same order as nodes)

    Returns:
        groups: list of lists, each inner list is one partition set
    """
    groups = {}
    for node, group_id in zip(nodes, chromosome):
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(node)
    return list(groups.values())


def check_partition(groups, edges):
    """
    Validates that no edge exists within any partition group.
    Used by the RL environment as the validity checker in Phase 1.
    Invalid actions are masked — the agent never sees them.

    Args:
        groups: list of lists of nodes
        edges : list of edges as sorted tuples (u, v)

    Returns:
        True if valid, False if any edge found within a group
    """
    edge_set = set(tuple(sorted(e)) for e in edges)
    for group in groups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                u = group[i]
                v = group[j]
                if tuple(sorted((u, v))) in edge_set:
                    return False
    return True


if __name__ == "__main__":
    from fixed_graph_generation import generate_large_network
    nodes, edges, sessions = generate_large_network()

    print("Testing 30 random valid partitions:\n")
    for i in range(30):
        chromosome = generate_random_valid_partition(nodes, edges)
        groups     = decode_partition(nodes, chromosome)
        valid      = check_partition(groups, edges)
        print(f"Run {i+1:02d} | Groups: {len(groups):3d} | Valid: {valid}")
