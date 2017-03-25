import numpy as np
#Adapted from James Clough
from log_bin import log_bin, lin_bin
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from scipy import optimize
from scipy import integrate
import os.path

font = {'family' : 'Arial',
        'size'   : 16}
matplotlib.rc('font', **font)
cm = plt.get_cmap('brg')

method=1
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
    k_list = dist_file[:,0]
    frequency = dist_file[:,1]
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
    prob_mean = prob_mean[m:]
    prob_stderr = prob_stderr[m:]
    k = k[m:]
    np.save(path,np.array([k,prob_mean,prob_stderr]))
    return k, prob_mean, prob_stderr

def get_k_prob_dist_log(m,N,runs, a=1.25):
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
    k = np.array(k,dtype='float')
    if method == 0 or (method == 2 and L>0):
        return float(2*m*(m+1))/(k*(k+1.)*(k+2.))
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


def plot_k_dist(m,N,runs, process='logbin', err=True,mark='.-',color='r',lw=1.5):
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
    
    if N==1e7:
        plt.plot(k_theory, prob_theory,'k--', lw=2, label='Theory')
    if err or process=='raw':
        errs = plt.errorbar(k, prob,stderr,fmt=mark,c=color, 
                        lw=lw, errorevery=1, markersize=5,
                        label='m='+str(m))
        errs[-1][0].set_linestyle('--')
        errs[-1][0].set_linewidth(1)
    else:
        plt.plot(k, prob,ls=mark,c=color, 
                        lw=lw, markersize=5,
                        label='N=1e'+str(int(np.log10(N))))
    plt.xscale("log",nonposx='clip')
    plt.yscale("log",nonposy='clip')
    plt.xlabel('$k$')
    plt.ylabel('$p(k)$')
    plt.legend(loc=0)
    plt.tight_layout()
    plt.show()


def plot_dist_multiple_m(m_max,N,runs):
    markers = ['+-','x-','|-','.-']
    colors = ['m','g','r','b']
    m_list = [20,5,2,1]
    for i in xrange(len(m_list)):
        plot_k_dist(m_list[i],N,runs,color=colors[i],mark=markers[i])
    
def plot_dist_raw_log(m,N,runs):
    plot_k_dist(m,N,runs,'raw',color='b',mark='.',err=False,lw=1)
    plot_k_dist(m,N,runs,'logbin',color='r',mark='.-',lw=2)
    
def plot_dist_multiple_N(m,N_max,runs):
    N_list = [1e5,1e4,1e3,1e2]
    colors = [cm(1.*i/6) for i in range(6)]
    for i in xrange(len(N_list)):
        plot_k_dist(m,N_list[i],runs,color=colors[i],mark='-',lw=2,err=False)

def plot_k_dist_walker(m,N,runs,L_max, process='logbin'):
    max_k = 0
    slope_list = []
    L_list = [0,1,2,5,10]
    global L
    plt.figure(0)
    colors = [cm(1.*i/len(L_list)) for i in range(len(L_list))]
    for i in xrange(len(L_list)):
        L = L_list[i]
        if process == 'logbin':
            k, prob, stderr = get_k_prob_dist_log(m,N,runs)
            ls='-'
        elif process == 'raw':
            k, prob, stderr = get_k_prob_dist(m,N,runs)
            ls = '.'
        if max_k < np.max(k):
            max_k = np.max(k)
        
        if L>0:
            end = 20
            start = 10
            #print k[start]
            #print k[end]
            slope = np.polyfit(np.log(k[start:end]),np.log(prob[start:end]),1)[0]
            #print slope
            slope_list.append(slope)                
        plt.plot(k, prob,ls=ls, lw=2, c=colors[i],
                     label='$\ell =$'+str(L_list[i]))
        
    k_theory, prob_theory = gen_theoretical_dist(m,np.max(k))
    plt.plot(k_theory, prob_theory,'k--', lw=1.5, label='Theory')
    plt.xscale("log",nonposx='clip')
    plt.yscale("log",nonposy='clip')
    plt.xlabel('$k$')
    plt.ylabel('$p(k)$')
    plt.legend(loc=3,ncol=2)
    plt.show()
    
    plt.figure(1)
    plt.plot(L_list[1:],-1*np.array(slope_list), 'b-',
             label='Observed', lw=1.5)
    plt.plot([1,20],[3,3],'k--',lw=2, label='Expected')
    plt.xlabel('$\ell$')
    plt.ylabel('$\\alpha$ $p(k) \propto k^{-\\alpha}$')
    plt.legend(loc=4)
    plt.xticks(L_list)
    plt.tight_layout()
    plt.show()
        





def cdf_0(k_vals,m):
    vals = []
    for k_val in k_vals:
        k = np.arange(m,k_val+1)
        dist = float(2*m*(m+1))/(k*(k+1)*(k+2))
        vals.append(np.sum(dist))
    return np.array(vals)

def cdf_1(k_vals,m):
    vals = []
    for k_val in k_vals:
        k = np.arange(m,k_val+1, dtype='float')
        dist = 1./float(m+1) * (float(m)/float(m+1))**(k-m)
        vals.append(np.sum(dist))
    return np.array(vals)


def ks_test_dist(m,N,runs):
    k, prob, stderr = get_k_prob_dist(m,N,runs)
    k,prob,stderr = k[:],prob[:],stderr[:]   
    
    
    norm_prob = prob/np.sum(prob)
    d = np.random.choice(k,size=int(1e6), p = norm_prob)    
    
    def ecdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs)+1)/float(len(xs))
        return xs, ys

    xs,ys = ecdf(d)
    ind = np.cumsum(np.bincount(np.array(xs,dtype='int')))[m:] - 1

    if method == 0:
        y = cdf_0(xs,m)
    elif method == 1:
        y = cdf_1(xs,m)

    diff = np.abs(y-ys)
    D = np.max(diff[ind])
    D_ind = np.where(diff==D)
    print 'D statistic =',D
    #print xs[D_ind]
    print 'Critical D for 20% significance =', 1.07/np.sqrt(d.size)
    plt.plot(xs,ys,'r',lw=1.5, label='Empirical CDF')
    plt.plot(xs,y,'b',lw=1.5, label='Theoretical CDF')
    plt.xlabel('$k$')
    plt.ylabel('Cumulative Probability')
    plt.legend(loc=4)
    plt.tight_layout()
    
    
    

def chi_squared_test(m,N,runs, tail=1e5):
    k, prob, stderr = get_k_prob_dist(m,N,runs)
    k,prob,stderr = k[:tail],prob[:tail],stderr[:tail]
    prob_theory = k_dist_theory(m,k)
    
    #print prob
    #print prob_theory
    #print N*runs*np.sum((prob - prob_theory)*(prob - prob_theory)/prob)
    #x = N*runs*prob
    #print len(x[x<5])
    #z = N*runs*prob_theory
    #print len(z[z<5])
    
    #reduced_chi = np.sum((prob-prob_theory)*(prob-prob_theory)/(stderr*stderr))
    #print reduced_chi
    
    results2 = stats.chisquare(N*runs*prob, N*runs*prob_theory, ddof=0)
    return results2

def plot_chi_squared(m,N,runs,start,end):
    p = []
    taildata = range(start,end,1)
    for tail in taildata:
        p_value = chi_squared_test(m,N,runs,tail)[1]
        p.append(p_value)
    plt.plot(taildata, p, 'b.')
    plt.xlabel('Cutoff Degree')
    plt.ylabel('p-value')
    plt.tight_layout()
    plt.show()


def two_sample_test(m,N,runs,times=10, alpha=0.9, tail=-1):
    k, prob, stderr = get_k_prob_dist(m,N,runs)
    k,prob,stderr = k[:tail],prob[:tail],stderr[:tail]
    k_theory = np.arange(m,int(1e6),dtype='float')
    prob_theory = k_dist_theory(m,k_theory)
    
    p_values = np.zeros(times)
    for i in xrange(times):
        sample_calc = np.random.choice(k_theory,int(1e5), p = prob_theory/np.sum(prob_theory)) 
        sample_obs = np.random.choice(k,int(1e5), p = prob/np.sum(prob))
    
        results = stats.ks_2samp(sample_calc,sample_obs)
        p_values[i]=results[1]
    accept_percent = p_values[p_values>alpha].size/float(p_values.size)
    return accept_percent
    



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
        plt.errorbar(10**log_N, log_mean, yerr=std_list, lw=1.5, label='Observed')
        plt.plot(10**log_N, expected, 'r--', lw=2, label='Theory')
        plt.xlabel('$N$')
        plt.ylabel('$k_1$')
        plt.legend(loc=2)
        plt.show()    
    return (power, powerErr), (prop,propErr)

def find_prop_k_max(m_max, N_max):
    m_list = np.arange(1,m_max+1, dtype='float')
    #m_list = np.array([1.,2.,3.,4.,5.,10.,15.,20.])
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
    
    plt.errorbar(m_list, prop_list, yerr=prop_err_list, label='Observed', lw=2)
    if method == 0:
        expected = np.sqrt(2*m_list*(m_list+1))
    if method == 1:
        expected = 1./np.log10((m_list+1)/m_list)
    plt.plot(m_list, expected, label='Expected',lw=2)
    plt.xlabel('$m$')
    plt.ylabel('Prop. Factor')
    plt.xticks(m_list)
    plt.legend(loc=2)
    plt.tight_layout()
    plt.show()


def plot_dist_collapsed(m=3, N_max=1e7, runs=1000):
    colors = [cm(1.*i/6) for i in range(6)]
    if method == 0:
        N_range = 10**np.arange(2,int(np.log10(N_max)+1), dtype='float')
        N_range = N_range[::-1]
        for i in xrange(len(N_range)):
            k, prob, stderr = get_k_prob_dist_log(m,N_range[i],runs)
            scaled_k = np.divide(k,np.sqrt(N_range[i]))
            scaled_prob = np.multiply(prob/(2*m*(m+1)), k*(k+1)*(k+2))
            
            plt.loglog(scaled_k, scaled_prob,'-', lw=1.5, c= colors[i],
                       label="N=1e"+str(int(np.log10(N_range[i]))))
            plt.xlabel('$k/N^D$')
            plt.ylabel('$p(k)/p_{theory}(k)$')
        plt.legend(loc=0)
    if method == 1:
        N_range = 10**np.arange(2,int(np.log10(N_max)+1), dtype='float')
        N_range = N_range[::-1]
        for i in xrange(len(N_range)):
            k, prob, stderr = get_k_prob_dist_log(m,N_range[i],runs)
            scaled_k = np.divide(k,np.log(N_range[i]))
            scaled_prob = np.divide(prob, k_dist_theory(m,k))
            
            plt.loglog(scaled_k, scaled_prob,'-', lw=1.5, c= colors[i],
                       label="N=1e"+str(int(np.log10(N_range[i]))))
            plt.xlabel('$k/ln(N)$')
            plt.ylabel('$p(k)/p_{theory}(k)$')
        plt.legend(loc=0)
        
        