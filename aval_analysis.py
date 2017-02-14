import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from oslo import Oslo
from log_bin import log_bin
font = {'family' : 'Arial',
        'size'   : 16}
matplotlib.rc('font', **font)
cm = plt.get_cmap('nipy_spectral')

sys_sizes = [8,16,32,64,128,256,512,1024]
sys_sizes = [8,16,32,64,128,256,512,1024,2048]


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

def gen_aval_prob_lin(L):
    s_list = gen_aval_list(L)
    time = float(len(s_list))
    s_hist = np.histogram(s_list, np.arange(np.min(s_list),
                                    np.max(s_list)+2,1))
    s_prob = s_hist[0]/float(time)
    s_range = s_hist[1][:-1]
    return s_prob, s_range

def gen_aval_prob_log(L):
    s_list = gen_aval_list(L)
    s_range, s_prob = log_bin(s_list,0,1,1.1,'integer')
    return s_prob, s_range


def plot_aval_prob(bin_type='log', task='3b'):
    """
    bin_type - 'lin', 'log', or 'both'.
    """
    prob_dist, range_list = [], []
    if bin_type == 'both':
        prob_dist_lin, range_list_lin = [], []
    for size in sys_sizes:
        if bin_type == 'lin':        
            s_prob, s_range = gen_aval_prob_lin(size)
        elif bin_type == 'log':
            s_prob, s_range = gen_aval_prob_log(size)
        elif bin_type == 'both':
            s_prob, s_range = gen_aval_prob_log(size)
            s_prob_lin, s_range_lin = gen_aval_prob_lin(size)
            prob_dist_lin.append(s_prob_lin)
            range_list_lin.append(s_range_lin)
        prob_dist.append(s_prob)
        range_list.append(s_range)
    
    if task=='3a':
        for i in xrange(len(sys_sizes)):
            if bin_type == 'both':
                plt.loglog(range_list_lin[i], prob_dist_lin[i],
                           'b.', label='Raw')
            plt.loglog(range_list[i],prob_dist[i], 
                           'r-', label='Log Binned', lw=2)
            plt.xlabel('Avalanche Size $s$ (# of grains toppled)')
            
    if task=='3b':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_color_cycle([cm(1.*i/9) for i in range(9)]) 
        for i in xrange(len(sys_sizes)):
            if bin_type == 'both':
                plt.loglog(range_list_lin[i], prob_dist_lin[i],
                           '.', label='L = '+str(sys_sizes[i]))
            ax.loglog(range_list[i],prob_dist[i], 
                           '-', label='L = '+str(sys_sizes[i]), lw=2)
            plt.xlabel('Avalanche Size $s$')
    
    plt.ylabel('Avalanche Size Probability $P_N(s;L)$')
    plt.legend(ncol=2, loc=1)
    plt.show()


#TASK 3c
def plot_aval_prob_collapsed(D = 2.252, tau = 1.557):
    print D*(2. - tau)
    prob_dist, range_list = [], []
    for size in sys_sizes:
        s_prob, s_range = gen_aval_prob_log(size)
        prob_dist.append(s_prob)
        range_list.append(s_range)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/9) for i in range(9)])    
    for i in xrange(len(sys_sizes)):
        scaled_prob = np.multiply(np.array(range_list[i])**float(tau),
                                                          prob_dist[i])
        scaled_range = np.divide(range_list[i], 
                                 float(sys_sizes[i]**float(D)))
        
        ax.loglog(scaled_range,scaled_prob,'-', lw=1.5,
                  label='L = '+str(sys_sizes[i]))
    
    plt.xlabel('Scaled Avalanche Size $s/L^D$')
    plt.ylabel('Scaled Avalanche Size Probability $s^{\\tau_s} P_N(s;L)$')
    plt.legend(loc=3, ncol=2)
    plt.show()


#TASK 3d
def calc_kth_moment(k, L):
    s_list = gen_aval_list(L)
    time = float(len(s_list))
    kth_moment = np.sum(np.array(s_list)**float(k))/float(time)
    return kth_moment    
    
def plot_moments(k_max=5):
    k_range= range(1,k_max+1)
    moment_list = []
    param_list = []
    for k in k_range:
        moments = []
        for size in sys_sizes:
            moments.append(calc_kth_moment(k, size))
        param = np.polyfit(np.log(sys_sizes)[-4:], np.log(moments)[-4:], 1)
        moment_list.append(moments)
        param_list.append(param)
    
    for i in range(k_max):
        plt.loglog(sys_sizes, moment_list[i], 'k.', lw=2)
        fit = np.exp(param_list[i][1])*np.array(sys_sizes)**float(param_list[i][0])
        plt.loglog(sys_sizes, fit, 'b-', lw=1.5)
    
    plt.xlabel('System Size L')
    plt.ylabel('kth Moment $\overline{s^k}$')
    plt.show()
    
    
    
def moment_size_scaling(k, plot_scaling=True):
    moments = []
    for size in sys_sizes:
        moments.append(calc_kth_moment(k, size))

    param = np.polyfit(np.log(sys_sizes)[-4:], np.log(moments)[-4:], 1)
    if plot_scaling:
        print param
        scaled_moments = np.divide(moments,
                        np.exp(param[1])*np.array(sys_sizes)**float(param[0]))
        scal_param = np.polyfit(np.log(sys_sizes)[-4:], 
                             np.log(scaled_moments)[-4:], 1)
        scaled_moments -= 1.
        fit = scal_param[0]*np.log(sys_sizes) + scal_param[1]
        fit = np.exp(fit) - 1.
        thresh = np.repeat(0.01,len(sys_sizes))
        plt.figure()
        plt.semilogx(sys_sizes, scaled_moments, '.', label='Data', lw=2)
        plt.semilogx(sys_sizes, fit, 'g-', label='Fit' , lw=2)
        plt.semilogx(sys_sizes, thresh, 'r--', label='1% Threshold')
        plt.xlim(0, sys_sizes[-1])
        plt.ylim(-np.max(scaled_moments)*0.1,np.max(scaled_moments)*1.1)
        plt.xlabel('System Size L')
        plt.ylabel('Scaling Error')
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
    plt.plot(k_range, slope_list, 'k.', label='Data')
    plt.plot(k_range, fit, 'b-',label='Fit')
    plt.xlabel('$k$')
    plt.ylabel('$D(1+k-\\tau_s)$')
    plt.legend(loc=0)
    plt.show()
        
