import numpy as np
from log_bin import log_bin
import matplotlib.pyplot as plt


def load_edges(m,N):
    edge_list = np.loadtxt("./data/edgelist_"+str(int(m))+"_"+str(int(N))+".txt",
                           dtype='int')
    return edge_list

def load_dist(m,N):
    dist_file = np.loadtxt("./data/degreedist_"+str(int(m))+"_"+str(int(N))+".txt",
                            skiprows=1, dtype='int')
    k_list = dist_file[:,0]
    frequency = dist_file[:,1]
    return k_list, frequency

def get_k_prob_dist(m,N):
    k, freq = load_dist(m,N)
    prob = np.divide(freq,float(np.sum(freq)))
    return k, prob

def get_k_prob_dist_log(m,N):
    k_list, freq = load_dist(m,N)
    raw_data = np.repeat(k_list,freq)
    k, prob = log_bin(raw_data,m,1.,1.5)
    return k, prob

def gen_theoretical_dist(m,k_max):
    k = np.arange(m,k_max+1, dtype='float')
    dist = float(2*m*(m+1))/(k*(k+1)*(k+2))
    return k, dist
    

def plot_k_dist(m,N):
    k, prob = get_k_prob_dist_log(m,N)
    k_theory, prob_theory = gen_theoretical_dist(m,np.max(k))
    print prob
    plt.loglog(k, prob,'-', lw=2, label='Data')
    plt.loglog(k_theory, prob_theory,'--', lw=1.5, label='Theory')
    plt.xlabel('$k$')
    plt.ylabel('$p(k)$')
    plt.legend(loc=0)
    plt.show()