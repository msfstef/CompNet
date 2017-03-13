import numpy as np
from log_bin import log_bin, lin_bin
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize
from scipy import integrate
import os.path

method=2
L=15


def load_edges(m,N):
    edge_list = np.loadtxt("./data/edgelist_"+str(int(m))+"_"+str(int(N))+".txt",
                           dtype='int')
    return edge_list

def load_dist_run(m,N,run):
    path = str("./data/degreedistrun_"+str(int(m))+"_"+str(int(N))+
                            "_"+str(int(run))+"_"+str(method))
    if method == 2:
        path += str("_" + str(L))
    dist_file = np.loadtxt(path+".txt", skiprows=1, dtype='int')
    k_list = dist_file[1:,0]
    frequency = dist_file[1:,1]
    return k_list, frequency

def load_k_max(m,N):
    path = str("./data/kmax_"+str(int(m))+"_"+str(int(N))+
                            "_"+str(method))
    if method == 2:
        path += str("_" + str(L))
    k_max_file = np.loadtxt(path+".txt", skiprows=1, dtype='int')
    return k_max_file

def get_k_prob_dist(m,N,runs):
    path = str("./data/degreedistraw_"+str(int(m))+"_"+str(int(N))+
                            "_"+str(int(runs))+"_"+str(method))
    if method == 2:
        path += str("_"+str(L))
    path += str(".npy")
    if os.path.isfile(path):
        data = np.load(path)
        return data[0], data[1], data[2]
    k_list = np.empty(runs,dtype='object')
    freq_list= np.empty(runs,dtype='object')
    for run in range(runs):
        k, freq = load_dist_run(m,N,run)
        k_list[run] = k
        freq_list[run] = np.array(freq,dtype='float')
    k = np.array(max(k_list, key=len), dtype='float')
    for i in xrange(len(freq_list)):
        if len(k)>len(freq_list[i]):
            freq_list[i]=np.pad(freq_list[i],(0,len(k)-len(freq_list[i])),'constant')
    
    prob_mean = np.mean(freq_list,axis=0)/float(N)
    prob_stderr = np.std(freq_list,axis=0)/float(N)
    np.save(path,np.array([k,prob_mean,prob_stderr]))
    return k, prob_mean, prob_stderr

def get_k_prob_dist_log(m,N,runs, a=1.3):
    path = str("./data/degreedistlogbin_"+str(int(m))+"_"+str(int(N))+
                            "_"+str(int(runs))+"_"+str(method))
    if method == 2:
        path += str("_"+str(L))
    path += str(".npy")
    
    if os.path.isfile(path):
        data = np.load(path)
        return data[0], data[1], data[2]
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
    np.save(path,np.array([k,prob_mean,prob_stderr]))
    return k, prob_mean, prob_stderr
    
def get_k_prob_dist_cdf(m,N,runs):
    k, prob = get_k_prob_dist(m,N,runs)
    cdf = np.cumsum(prob)
    return k, cdf


def k_dist_theory(m, k):
    if method == 0 or (method == 2 and L>0):
        return float(2*m*(m+1))/(k*(k+1)*(k+2))
    elif method == 1 or method == 2:
        A = 1./float((m+1))
        return A*np.power(m*A,k-float(m))

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


def plot_k_dist(m,N,runs, process='logbin', err=True):
    if process == 'logbin':
        k, prob, stderr = get_k_prob_dist_log(m,N,runs)
        k_theory, prob_theory = gen_theoretical_dist(m,np.max(k))
        ls='-'
    elif process == 'cdf':
        k, prob = get_k_prob_dist_cdf(m,N,runs)
        k_theory, prob_theory = gen_theoretical_cdf(m,np.max(k))
        stderr = np.zeros(len(prob))
        ls = '-'
    elif process == 'raw':
        k, prob, stderr = get_k_prob_dist(m,N,runs)
        k_theory, prob_theory = gen_theoretical_dist(m,np.max(k))
        ls = '.'
    if not err:
        stderr = np.zeros(len(prob))
    plt.errorbar(k, prob,stderr,fmt=ls, lw=2, label='Data')
    plt.plot(k_theory, prob_theory,'--', lw=1.5, label='Theory')
    plt.xscale("log",nonposx='clip')
    plt.yscale("log",nonposy='clip')
    plt.xlabel('$k$')
    plt.ylabel('$p(k)$')
    plt.legend(loc=0)
    plt.show()


def func(k_vals, m=1.):
    vals = []
    def dist(k):
        return float(2*m*(m+1))/(k*(k+1)*(k+2))
    for k in k_vals:
        vals.append(integrate.quad(dist,float(m),float(k))[0])
    return np.array(vals)
    
def func2(k_vals,m=1):
    vals = []
    for k_val in k_vals:
        k = np.arange(m,k_val+1)
        dist = float(2*m*(m+1))/(k*(k+1)*(k+2))
        vals.append(np.sum(dist))
    return np.array(vals)


def ks_test_dist(m,N,runs):
    k, prob, stderr = get_k_prob_dist(m,N,runs)
    k,prob,stderr = k[:],prob[:],stderr[:]
    k_theory = np.arange(m,int(1e7),dtype='float')
    prob_theory = k_dist_theory(m,k_theory)
    print prob_theory/np.sum(prob_theory)
    
    c = np.random.choice(k,10000, p = prob/np.sum(prob))
    d = np.random.choice(k_theory,10000, p = prob_theory/np.sum(prob_theory))
    print max(d)
    print max(c)
    
    #stderr = stderr/prob
    #prob = np.log10(prob)
    #prob_theory = np.log10(prob_theory)
    
    #reduced_chi = np.sum((prob-prob_theory)*(prob-prob_theory)/(stderr*stderr))
    #print reduced_chi
    
    kstest = stats.kstest(c, func2)    
    print kstest
    
    results = stats.ks_2samp(c,d)
    print results
    results2 = stats.chisquare(N*runs*prob, N*runs*prob_theory)
    print results2




def fit_k_max_vs_N(m, N_max, plot=True):
    mean_list = []
    std_list = []
    exp_list = range(2,int(np.log10(N_max))+1)
    for i in exp_list:
        k_list = load_k_max(m,int(10**float(i)))
        mean_list.append(np.mean(k_list))
        std_list.append(np.std(k_list))
    
    if method == 0:
        log_mean = np.log10(mean_list)
        log_std = np.array(std_list)/np.array(mean_list)   
    elif method == 1:
        log_mean = np.array(mean_list, dtype='float')
        log_std = np.array(std_list, dtype='float')
    
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
        plt.xscale("log")
        expected = power*log_N + np.log10(prop)
        if method == 0:
            plt.yscale("log")
            log_mean = 10**log_mean
            expected = prop*10**(power*log_N)
        plt.errorbar(10**log_N, log_mean, yerr=std_list,label='Data')
        plt.plot(10**log_N, expected, 'g--', label='Fit')
        plt.show()    
    return (power, powerErr), (prop,propErr)

def find_prop_k_max(m_max, N_max):
    m_list = np.arange(1,m_max+1, dtype='float')
    prop_list = []
    prop_err_list = []
    for m in m_list:
        if method == 0:
            prop = fit_k_max_vs_N(m,N_max,False)[1]
            print fit_k_max_vs_N(m,N_max,False)[0]
        elif method == 1:
            prop = fit_k_max_vs_N(m,N_max,False)[0]
            print fit_k_max_vs_N(m,N_max,False)[0]
        prop_list.append(prop[0])
        prop_err_list.append(prop[1])
    
    plt.errorbar(m_list, prop_list, yerr=prop_err_list)
    if method == 0:
        expected = np.sqrt(2*m_list*(m_list+1))
    if method == 1:
        expected = 1./np.log10((m_list+1)/m_list)
    plt.plot(m_list, expected)
    plt.show()


def plot_dist_collapsed(D=0.5, m=3, N_max=1e7, runs=500):
    N_range = 10**np.arange(2,int(np.log10(N_max)+1), dtype='float')
    for N in N_range:
        k, prob, stderr = get_k_prob_dist_log(m,N,runs)
        scaled_k = np.divide(k,np.sqrt(N))
        scaled_prob = np.multiply(prob/(2*m*(m+1)), k*(k+1)*(k+2))
        
        plt.loglog(scaled_k, scaled_prob,'-', lw=2, label=np.log10(N))
        plt.xlabel('$k/N^D$')
        plt.ylabel('$p(k)/p_{theory}(k)$')
    plt.legend(loc=0)
        
        