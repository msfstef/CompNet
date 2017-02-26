import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



N=int(1e3)
m = 4

def gen_graph(m, N):
    G=nx.complete_graph(m)
    for i in xrange(m,N):
        G.add_node(i)
        vertex_list = np.array(G.edges()).flatten()
        ends = np.random.choice(vertex_list, m)
        while len(ends) != len(set(ends)):
            ends = np.random.choice(vertex_list, m)
        for j in xrange(0,m):        
            G.add_edge(i, ends[j])
    return G


G = gen_graph(m,N)
print len(G.nodes())
print len(G.edges())

