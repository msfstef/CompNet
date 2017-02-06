import numpy as np
import matplotlib.pyplot as plt
from oslo import Oslo
from log_bin import log_bin

sys_sizes = [8,16,32,64,128,256]
sys_sizes = [8,16,32,64,128,256,512,1024,2048]

# BONUS TASK 1
def gen_drop_list(L, time=1e7, gen=False, save=True):
    d_list = []
    if not gen:
        print 'Size', L,'completed (max', sys_sizes[-1],').'
        return np.load('dlist'+str(L)+'.npy')
    sys = Oslo(L)
    sys.simulate(L*L)
    
    for i in xrange(int(time)):
        sys.simulate(1)
        if sys.drop_size > 0:
            d_list.append(sys.drop_size)
        
    if save:
        np.save('dlist'+str(L)+'.npy',np.array(d_list))
    print 'Size', L,'completed (max', sys_sizes[-1],').'
    return d_list


def gen_drop_prob(L):
    d_list = gen_drop_list(L)
    d_hist = np.histogram(d_list, np.arange(np.min(d_list),
                                    np.max(d_list)+2,1))
    d_prob = d_hist[0]/float(len(d_list))
    d_range = d_hist[1][:-1]
    return d_prob, d_range


def gen_drop_prob_log(L):
    d_list = gen_drop_list(L)
    d_range, d_prob = log_bin(d_list,0,1,1.1,'integer')
    return d_prob, d_range


def plot_drop_prob():
    prob_dist, range_list = [], []
    for size in sys_sizes:
        d_prob, d_range = gen_drop_prob_log(size)
        prob_dist.append(d_prob)
        range_list.append(d_range)
    
    for i in range(len(sys_sizes)):
        plt.loglog(range_list[i],prob_dist[i], label=sys_sizes[i])
    plt.legend()
    plt.show()


def plot_drop_prob_collapsed(D = 1.2, tau = 1.02):
    prob_dist, range_list = [], []
    for size in sys_sizes:
        d_prob, d_range = gen_drop_prob_log(size)
        prob_dist.append(d_prob)
        range_list.append(d_range)
    
    for i in xrange(len(sys_sizes)):
        scaled_prob = np.multiply(np.array(range_list[i])**float(tau),
                                                          prob_dist[i])
        scaled_range = np.divide(range_list[i], 
                                 float(sys_sizes[i]**float(D)))
        
        plt.loglog(scaled_range,scaled_prob,'-', label=sys_sizes[i])
    plt.legend(loc=3)
    plt.show()


def plot_drop_prob_collapsed_alt(D_1 = 1.5, D_2 = 1.2, tau = 0.23):
    prob_dist, range_list = [], []
    for size in sys_sizes:
        d_prob, d_range = gen_drop_prob_log(size)
        prob_dist.append(d_prob)
        range_list.append(d_range)
    
    for i in xrange(len(sys_sizes)):
        scaled_prob = np.divide(prob_dist[i], 
                                np.array(range_list[i])**float(tau))
        scaled_prob = np.multiply(scaled_prob, sys_sizes[i]**float(D_1))
        scaled_range = np.divide(range_list[i], 
                                 float(sys_sizes[i]**float(D_2)))
        
        plt.loglog(scaled_range,scaled_prob,'-', label=sys_sizes[i])
    plt.legend(loc=3)
    plt.show()

#plot_drop_prob_collapsed()
#plot_drop_prob_collapsed_alt()