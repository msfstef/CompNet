import numpy as np
from log_bin import log_bin, lin_bin
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize

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

def load_k_max(m,N):
    k_max_file = np.loadtxt("./data/kmax_"+str(int(m))+"_"+str(int(N))+".txt",
                            skiprows=1, dtype='int')
    return k_max_file

def get_k_prob_dist(m,N):
    k, freq = load_dist(m,N)
    prob = np.divide(freq,float(np.sum(freq)))
    return k, prob

def get_k_prob_dist_log(m,N):
    k_list, freq = load_dist(m,N)
    raw_data = np.repeat(k_list,freq)
    k, prob = log_bin(raw_data,m,1.,1.2)
    return k, prob
    
def get_k_prob_dist_cdf(m,N):
    k, prob = get_k_prob_dist(m,N)
    cdf = np.cumsum(prob)
    return k, cdf


def k_dist_theory(m, k):
    return float(2*m*(m+1))/(k*(k+1)*(k+2))

def gen_theoretical_dist(m,k_max):
    k = np.arange(m,k_max+1, dtype='float')
    return k, k_dist_theory(m,k)

def gen_theoretical_cdf(m,k_max):
    k, dist = gen_theoretical_dist(m,k_max)
    cdf = np.cumsum(dist)
    return k, cdf
        


def plot_k_max_dist(m,N, bins=10):
    k_max_data = load_k_max(m,N)
    k_max, freq = lin_bin(k_max_data, bins)
    plt.plot(k_max, freq, lw=2)
    plt.show()


def plot_k_dist(m,N, method='logbin'):
    if method == 'logbin':
        k, prob = get_k_prob_dist_log(m,N)
        k_theory, prob_theory = gen_theoretical_dist(m,np.max(k))
        ls='-'
    elif method == 'cdf':
        k, prob = get_k_prob_dist_cdf(m,N)
        k_theory, prob_theory = gen_theoretical_cdf(m,np.max(k))
        ls = '-'
    elif method == 'raw':
        k, prob = get_k_prob_dist(m,N)
        k_theory, prob_theory = gen_theoretical_dist(m,np.max(k))
        ls = '.'
    plt.loglog(k, prob,ls, lw=2, label='Data')
    plt.loglog(k_theory, prob_theory,'--', lw=1.5, label='Theory')
    plt.xlabel('$k$')
    plt.ylabel('$p(k)$')
    plt.legend(loc=0)
    plt.show()


def func(prob, m, k_d):
    vals = []
    k_vals = k_d[::-1]
    for k_val in k_vals:
        k = np.arange(m,k_val)
        dist = float(2*m*(m+1))/(k*(k+1)*(k+2))
        vals.append(np.sum(dist))
    return np.array(vals)

def ks_test_dist(m,N):
    k, prob = get_k_prob_dist_log(m,N)

    prob_theory = k_dist_theory(m,k)
    
    results = stats.ks_2samp(prob, prob_theory)
    print results
    results2 = stats.chisquare(prob, prob_theory)
    print results2






def fit_k_max_vs_N(m, N_max, plot=True):
    mean_list = []
    std_list = []
    exp_list = range(2,int(np.log10(N_max))+1)
    for i in exp_list:
        k_list = load_k_max(m,int(10**float(i)))
        mean_list.append(np.mean(k_list))
        std_list.append(np.std(k_list))
    
    log_mean = np.log10(mean_list)
    log_std = np.array(std_list)/np.array(mean_list)   
    log_N = np.array(exp_list, dtype='float')
    
    # From http://scipy-cookbook.readthedocs.io/items/FittingData.html#id2
    fitfunc = lambda p, x: p[1] * x + p[0]
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

    guess = [1.0, 0.5]
    out = optimize.leastsq(errfunc, guess,
                       args=(log_N, log_mean, log_std), full_output=1)

    pfinal = out[0]
    covar = out[1]
    
    power = pfinal[1]
    prop = 10.0**pfinal[0]
    
    powerErr = np.sqrt( covar[1][1] )
    propErr = np.sqrt( covar[0][0] ) * prop * np.log(10)
    
    if plot:
        plt.errorbar(10**log_N, 10**log_mean, yerr=10**log_std)
        plt.xscale("log")
        plt.yscale("log")
        plt.show()
    
    return (power, powerErr), (prop,propErr)

def find_prop_k_max(m_max, N_max):
    m_list = np.arange(3,m_max+1)
    intcp_list = []
    intcp_err_list = []
    for m in m_list:
        intcp = fit_k_max_vs_N(m,N_max,False)[1]
        print fit_k_max_vs_N(m,N_max,False)[0]
        intcp_list.append(intcp[0])
        intcp_err_list.append(intcp[1])
    
    plt.errorbar(m_list, intcp_list, yerr=intcp_list)
    plt.plot(m_list, np.sqrt(3*m_list))
    plt.show()


def plot_dist_collapsed(D=0.5, m=3, N_max=1e7):
    N_range = 10**np.arange(2,int(np.log10(N_max)), dtype='float')
    for N in N_range:
        k, prob = get_k_prob_dist_log(m,N)
        scaled_k = np.divide(k,np.sqrt(N))
        scaled_prob = np.multiply(prob/(2*m*(m+1)), k*(k+1)*(k+2))
        
        plt.loglog(scaled_k, scaled_prob,'-', lw=2, label=np.log10(N))
        plt.xlabel('$k/N^D$')
        plt.ylabel('$p(k)/p_{theory}(k)$')
    plt.legend(loc=0)
        
        