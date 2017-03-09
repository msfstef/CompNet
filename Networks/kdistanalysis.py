import numpy as np
from log_bin import log_bin, lin_bin
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize

method=1


def load_edges(m,N):
    edge_list = np.loadtxt("./data/edgelist_"+str(int(m))+"_"+str(int(N))+".txt",
                           dtype='int')
    return edge_list

def load_dist_run(m,N,run):
    dist_file = np.loadtxt("./data/degreedistrun_"+str(int(m))+"_"+str(int(N))+
                            "_"+str(int(run))+"_"+str(method)+".txt",
                            skiprows=1, dtype='int')
    k_list = dist_file[:,0]
    frequency = dist_file[:,1]
    return k_list, frequency

def load_k_max(m,N):
    k_max_file = np.loadtxt("./data/kmax_"+str(int(m))+"_"+str(int(N))+
                            "_"+str(method)+".txt", skiprows=1, dtype='int')
    return k_max_file

def get_k_prob_dist(m,N,runs):
    k_list = np.empty(1)
    freq_list = np.zeros(1)
    for run in range(runs):
        k, freq = load_dist_run(m,N,run)
        if len(k)>len(k_list):
            freq_list=np.pad(freq_list,(0,len(k)-len(k_list)),'constant')
            k_list = k
            freq_list += freq
    prob = np.divide(freq_list,float(np.sum(freq_list)))
    return k_list, prob

def get_k_prob_dist_log(m,N,runs, a=1.3):
    
    k_list = np.empty(runs,dtype='object')
    prob_list= np.empty(runs,dtype='object')
    for run in range(runs):
        k, freq = load_dist_run(m,N,run)
        raw_data = np.repeat(k,freq)
        k, prob = log_bin(raw_data, m, 1.,a)
        k_list[run] = k
        prob_list[run] = prob
    k = np.array(max(k_list, key=len))
    for i in xrange(len(prob_list)):
        if len(k)>len(prob_list[i]):
            prob_list[i]=np.pad(prob_list[i],(0,len(k)-len(prob_list[i])),'constant')
    
    prob_mean = np.mean(prob_list,axis=0)
    prob_stderr = np.std(prob_list,axis=0)
    return k, prob_mean, prob_stderr
    
def get_k_prob_dist_cdf(m,N,runs):
    k, prob = get_k_prob_dist(m,N,runs)
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


def plot_k_dist(m,N,runs, method='logbin'):
    if method == 'logbin':
        k, prob, stderr = get_k_prob_dist_log(m,N,runs)
        k_theory, prob_theory = gen_theoretical_dist(m,np.max(k))
        ls='-'
    elif method == 'cdf':
        k, prob = get_k_prob_dist_cdf(m,N,runs)
        k_theory, prob_theory = gen_theoretical_cdf(m,np.max(k))
        stderr = np.zeros(len(prob))
        ls = '-'
    elif method == 'raw':
        k, prob = get_k_prob_dist(m,N,runs)
        k_theory, prob_theory = gen_theoretical_dist(m,np.max(k))
        stderr = np.zeros(len(prob))
        ls = '.'
    plt.errorbar(k, prob,stderr,fmt=ls, lw=2, label='Data')
    plt.plot(k_theory, prob_theory,'--', lw=1.5, label='Theory')
    plt.xscale("log",nonposx='clip')
    plt.yscale("log",nonposy='clip')
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

def ks_test_dist(m,N,runs):
    k, prob, stderr = get_k_prob_dist_log(m,N,runs)
    k,prob,stderr = k[:20],prob[:20],stderr[:20]
    prob_theory = k_dist_theory(m,k)
    
    stderr = stderr/prob
    prob = np.log10(prob)
    prob_theory = np.log10(prob_theory)
    
    reduced_chi = np.sum((prob-prob_theory)*(prob-prob_theory)/(stderr*stderr))/float(len(prob))
    print reduced_chi
    
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
    m_list = np.arange(1,m_max+1)
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


def plot_dist_collapsed(D=0.5, m=3, N_max=1e7, runs=500):
    N_range = 10**np.arange(2,int(np.log10(N_max)), dtype='float')
    for N in N_range:
        k, prob, stderr = get_k_prob_dist_log(m,N,runs)
        scaled_k = np.divide(k,np.sqrt(N))
        scaled_prob = np.multiply(prob/(2*m*(m+1)), k*(k+1)*(k+2))
        
        plt.loglog(scaled_k, scaled_prob,'-', lw=2, label=np.log10(N))
        plt.xlabel('$k/N^D$')
        plt.ylabel('$p(k)/p_{theory}(k)$')
    plt.legend(loc=0)
        
        