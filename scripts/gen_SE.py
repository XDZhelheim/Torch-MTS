import numpy as np
import random
import pickle
import networkx as nx
from gensim.models import Word2Vec

IS_DIRECTED = True
P = 2
Q = 1
NUM_WALKS = 100
WALK_LENGTH = 80
WINDOW_SIZE = 10
ITER = 1000
SE_DIM = 64


class Graph:
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
                    )
                else:
                    prev = walk[-2]
                    next = cur_nbrs[
                        alias_draw(
                            alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1]
                        )
                    ]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # print("Walk iteration:")
        for walk_iter in range(num_walks):
            # print(str(walk_iter + 1), "/", str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(
                    self.node2vec_walk(walk_length=walk_length, start_node=node)
                )

        return walks

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]["weight"])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [
                G[node][nbr]["weight"] for nbr in sorted(G.neighbors(node))
            ]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs
            ]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype="int")

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def get_adjacency_matrix(adj_dist, normalized_k=0.1):
    # Calculates the standard deviation as theta.
    distances = adj_dist[~np.isinf(adj_dist)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(adj_dist / std))

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx


def gen_SE(adj_path, output_path):
    with open(adj_path, "rb") as f:
        adj = pickle.load(f)
    adj[adj == 0] = np.inf
    np.fill_diagonal(adj, 0)
    adj = get_adjacency_matrix(adj)

    graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    graph = Graph(graph, IS_DIRECTED, P, Q)
    graph.preprocess_transition_probs()
    walks = graph.simulate_walks(NUM_WALKS, WALK_LENGTH)

    walks = [list(map(str, walk)) for walk in walks]
    w2v = Word2Vec(
        walks, vector_size=SE_DIM, window=10, min_count=0, sg=1, workers=32, epochs=ITER
    )
    w2v.wv.save_word2vec_format(output_path)


if __name__ == "__main__":
    for dataset in ["PEMS03", "PEMS04", "PEMS07", "PEMS08", "PEMSD7M", "PEMSD7L"]:
        adj_path = f"../data/{dataset}/adj_{dataset}_distance.pkl"
        output_path = f"../data/{dataset}/SE_{dataset}.txt"

        gen_SE(adj_path, output_path)
        print(f"Finished {dataset} SE.")
