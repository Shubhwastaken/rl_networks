import random


def generate_large_network():
    """
    Fixed 7-node graph for training.

    Nodes:
      Sources      : S1, S2, S3
      Intermediate : v1
      Sinks        : t1, t2, t3

    Sessions:
      S1 -> t1
      S2 -> t2
      S3 -> t3

    Edges (10 total):
      S1 - S2   (source-source)
      S1 - v1   (source-intermediate)
      S2 - v1   (source-intermediate)
      S3 - v1   (source-intermediate)
      S1 - t3   (source-sink)
      S2 - t1   (source-sink)
      S3 - t2   (source-sink)
      v1 - t1   (intermediate-sink)
      v1 - t2   (intermediate-sink)
      v1 - t3   (intermediate-sink)

    Optimal partition from paper:
      P1 = {S1, S2}   I(P1,P1) = {}   (no session internal)
      P2 = {S3, v1}   I(P2,P2) = {}
      P3 = {t1, t2}   I(P3,P3) = {}
      P4 = {t3}       I(P4,P4) = {}

    Best known bound: |E| / (|I| + sum|I(Pi,Pi)|) = 10/6 = 1.67
    """

    nodes = ['S1', 'S2', 'S3', 'v1', 't1', 't2', 't3']

    edges = [
        ('S1', 'S2'),
        ('S1', 'v1'),
        ('S2', 'v1'),
        ('S3', 'v1'),
        ('S1', 't3'),
        ('S2', 't1'),
        ('S3', 't2'),
        ('t1', 'v1'),
        ('t2', 'v1'),
        ('t3', 'v1'),
    ]

    sessions = [('S1', 't1'), ('S2', 't2'), ('S3', 't3')]

    return nodes, edges, sessions


if __name__ == "__main__":
    nodes, edges, sessions = generate_large_network()

    print(f"Nodes   : {len(nodes)}  -> {nodes}")
    print(f"Edges   : {len(edges)}")
    print(f"Sessions: {len(sessions)}")
    print()
    print("Edge list:")
    for e in edges:
        print(f"  {e[0]} -- {e[1]}")
    print()
    print("Sessions:")
    for s, t in sessions:
        print(f"  {s} -> {t}")
    print()

    # compute trivial partition bound
    trivial = len(edges) / len(sessions)
    print(f"Trivial bound (no internal sessions): "
          f"{len(edges)}/{len(sessions)} = {trivial:.4f}")
    print(f"Target bound (optimal partition)    : "
          f"10/6 = {10/6:.4f}")