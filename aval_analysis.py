import numpy as np
import matplotlib.pyplot as plt
from oslo import Oslo
from log_bin import log_bin

sys_sizes = [8,16,32,64,128,256,512,1024]
#sys_sizes = [32,64,128,256,512,1024]

#TASK 3a and 3b
def gen_aval_list(L, time=1e8, gen=False, save=True):
    s_list = []
    if not gen:
        print 'Size', L,'completed (max', sys_sizes[-1],').'
        return np.load('slist'+str(L)+'.npy')
    sys = Oslo(L)
    sys.simulate(L*L)
    
    for i in xrange(int(time)):
        sys.simulate(1)
        s_list.append(sys.aval_size)
    
    if save:
        np.save('slist'+str(L)+'.npy',np.array(s_list))
    print 'Size', L,'completed (max', sys_sizes[-1],').'
    return s_list

def gen_aval_prob_lin(L, time = 1e8):
    s_list = gen_aval_list(L, time)
    s_hist = np.histogram(s_list, np.arange(np.min(s_list),
                                    np.max(s_list)+2,1))
    s_prob = s_hist[0]/float(time)
    s_range = s_hist[1][:-1]
    return s_prob, s_range

def gen_aval_prob_log(L, time = 1e8):
    s_list = gen_aval_list(L, time)
    s_range, s_prob = log_bin(s_list,0,1,1.1,'integer')
    return s_prob, s_range


def plot_aval_prob(time = 1e8, bin_type='log'):
    """
    bin_type - 'lin', 'log', or 'both'.
    """
    prob_dist, range_list = [], []
    if bin_type == 'both':
        prob_dist_lin, range_list_lin = [], []
    for size in sys_sizes:
        if bin_type == 'lin':        
            s_prob, s_range = gen_aval_prob_lin(size, time)
        elif bin_type == 'log':
            s_prob, s_range = gen_aval_prob_log(size, time)
        elif bin_type == 'both':
            s_prob, s_range = gen_aval_prob_log(size, time)
            s_prob_lin, s_range_lin = gen_aval_prob_lin(size, time)
            prob_dist_lin.append(s_prob_lin)
            range_list_lin.append(s_range_lin)
        prob_dist.append(s_prob)
        range_list.append(s_range)
    
    for i in xrange(len(sys_sizes)):
        plt.loglog(range_list[i],prob_dist[i], 
                   '.', label=sys_sizes[i])
        if bin_type == 'both':
            plt.loglog(range_list_lin[i], prob_dist_lin[i],
                       '.', label=sys_sizes[i])
    plt.legend()
    plt.show()


#TASK 3c
def plot_aval_prob_collapsed(D = 2.252, tau = 1.557):
    print D*(2. - tau)
    prob_dist, range_list = [], []
    for size in sys_sizes:
        s_prob, s_range = gen_aval_prob_log(size)
        prob_dist.append(s_prob)
        range_list.append(s_range)
    
    for i in xrange(len(sys_sizes)):
        scaled_prob = np.multiply(np.array(range_list[i])**float(tau),
                                                          prob_dist[i])
        scaled_range = np.divide(range_list[i], 
                                 float(sys_sizes[i]**float(D)))
        
        plt.loglog(scaled_range,scaled_prob,'.', label=sys_sizes[i])
    plt.legend(loc=3)
    plt.show()


#TASK 3d
def calc_kth_moment(k, L, time=1e6):
    s_list = gen_aval_list(L, time)
    kth_moment = np.sum(np.array(s_list)**float(k))/float(time)
    return kth_moment    
    
def moment_size_scaling(k, plot_scaling=True):
    moments = []
    for size in sys_sizes:
        moments.append(calc_kth_moment(k, size))

    param = np.polyfit(np.log(sys_sizes)[-4:], np.log(moments)[-4:], 1)
    if plot_scaling:
        print param
        scaled_moments = np.divide(moments,
                        np.array(sys_sizes)**float(2.252*(1+k-1.557)))
        scal_param = np.polyfit(np.log(sys_sizes)[-4:], 
                             np.log(scaled_moments)[-4:], 1)
        fit = scal_param[0]*np.log(sys_sizes) + scal_param[1]
        fit = np.exp(fit)
        plt.figure()
        plt.loglog(sys_sizes, scaled_moments, '.', label='Raw Data')
        plt.loglog(sys_sizes, fit, label='Fit')
        plt.xlim(0)
        plt.ylim(0,np.max(scaled_moments)*1.1)
        plt.legend(loc=1)
        plt.show()
    else:
        return param[0]


def moment_analysis(k_max):
    k_range = np.arange(1,k_max+1)
    slope_list = []
    for k in k_range:
        slope_list.append(moment_size_scaling(k,False))
    param = np.polyfit(k_range, slope_list, 1)
    print 'D:',param[0],'and tau:',1.-param[1]/param[0]
    print 'D(2-tau) =', (param[0]*(1 + param[1]/param[0]))
    fit = param[0]*k_range + param[1]
    plt.figure()
    plt.plot(k_range, slope_list, '.', label='Raw Data')
    plt.plot(k_range, fit, label='Fit')
    plt.legend(loc=2)
    plt.show()
        
