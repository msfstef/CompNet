import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from oslo import Oslo
from log_bin import log_bin
font = {'family' : 'Arial',
        'size'   : 16}
matplotlib.rc('font', **font)
cm = plt.get_cmap('nipy_spectral')

sys_sizes = [8,16,32,64,128,256]
sys_sizes = [8,16,32,64,128,256,512,1024,2048]

# BONUS TASK 1
def gen_drop_list(L, time=1e6, gen=True, save=False):
    """
    L - system size
    time - number of grains to add after reaching steady state
    gen - if set to True, generates new data from scratch. If set to False,
        loads npy data from the same folder.
    save - if set to True, saves data to npy files.
    
    Returns list of drop sizes after the system has reached the
    steady state for the given time and system size.
    """
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
    """
    Returns drop size probability distribution for the given system size L
    using a linear binning.
    """
    d_list = gen_drop_list(L)
    d_hist = np.histogram(d_list, np.arange(np.min(d_list),
                                    np.max(d_list)+2,1))
    d_prob = d_hist[0]/float(len(d_list))
    d_range = d_hist[1][:-1]
    return d_prob, d_range


def gen_drop_prob_log(L):
    """
    Returns drop size probability distribution for the given system size L
    using logarithmic binning from the log_bin module.
    """
    d_list = gen_drop_list(L)
    d_range, d_prob = log_bin(d_list,0.,1.,1.2,'integer')
    return d_prob, d_range


def plot_drop_prob():
    """
    Plots uncollapsed drop size probability distributions for the given
    system sizes sys_sizes.
    """
    prob_dist, range_list = [], []
    for size in sys_sizes:
        d_prob, d_range = gen_drop_prob_log(size)
        prob_dist.append(d_prob)
        range_list.append(d_range)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/9) for i in range(9)])
    for i in range(len(sys_sizes)):
        ax.loglog(range_list[i],prob_dist[i], 
                  label='L = '+str(sys_sizes[i]), lw=2)
    plt.xlabel('Drop Size $d$')
    plt.ylabel('Drop Size Probability $P(d;L)$')
    plt.xlim(0)
    plt.legend(loc=1, ncol=2)
    plt.show()


def plot_drop_prob_collapsed(D = 1.265, tau = 1.018):
    """
    Plots collapsed avalanche size probability distributions for the given
    system sizes sys_sizes using exponents D and tau as given.
    """
    print 'D =', D,'and tau =', tau 
    prob_dist, range_list = [], []
    for size in sys_sizes:
        d_prob, d_range = gen_drop_prob_log(size)
        prob_dist.append(d_prob)
        range_list.append(d_range)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/9) for i in range(9)])
    for i in xrange(len(sys_sizes)):
        scaled_prob = np.multiply(np.array(range_list[i])**float(tau),
                                                          prob_dist[i])
        scaled_range = np.divide(range_list[i], 
                                 float(sys_sizes[i]**float(D)))
        
        ax.loglog(scaled_range,scaled_prob,'-', 
                  label='L = '+str(sys_sizes[i]), lw=2)
    plt.xlabel('Scaled Drop Size $d/L^D$')
    plt.ylabel('Scaled Drop Size Probability $d^{\\tau_d}P(d;L)$')
    plt.xlim(0)
    plt.legend(loc=0, ncol=2)
    plt.show()


def plot_drop_prob_collapsed_alt(D_1 = 1.50, D_2 = 1.265, tau = 0.23):
    """
    Plots collapsed avalanche size probability distributions for the given
    system sizes sys_sizes using exponents D1, D2, and tau as given.
    """
    prob_dist, range_list = [], []
    for size in sys_sizes:
        d_prob, d_range = gen_drop_prob_log(size)
        prob_dist.append(d_prob)
        range_list.append(d_range)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/9) for i in range(9)])
    for i in xrange(len(sys_sizes)):
        scaled_prob = np.divide(prob_dist[i], 
                                np.array(range_list[i])**float(tau))
        scaled_prob = np.multiply(scaled_prob, sys_sizes[i]**float(D_1))
        scaled_range = np.divide(range_list[i], 
                                 float(sys_sizes[i]**float(D_2)))
        
        plt.loglog(scaled_range,scaled_prob,'-', label=sys_sizes[i])
    
    plt.xlabel('Scaled Drop Size $d/L^{D_2}$')
    plt.ylabel('Scaled Drop Size Probability $L^{D_1}d^{-\\tau_d}P(d;L)$')
    plt.xlim(0)
    plt.legend(loc=0, ncol=2)
    plt.show()
    

def calc_kth_moment(k, L):
    """
    Returns kth moment of avalanche size for the given system size L.
    """
    d_list = gen_drop_list(L)
    time = float(len(d_list))
    kth_moment = np.sum(np.array(d_list)**float(k))/float(time)
    return kth_moment    

    
def moment_size_scaling(k, plot_scaling=True):
    """
    Returns slope of kth moment vs system size if plot_scaling is set to False.
    Plots the corrections to scaling to the kth moment otherwise.
    """
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
    """
    Calculates kth moments against system size from k = 1 to k_max against
    system size and uses the slopes to estimate critical exponents D and tau.
    """
    k_range = np.arange(1,k_max+1)
    slope_list = []
    for k in k_range:
        slope_list.append(moment_size_scaling(k,False))
    param = np.polyfit(k_range, slope_list, 1)
    print 'D:',param[0],'and tau:',1.-param[1]/param[0]
    fit = param[0]*k_range + param[1]
    plt.figure()
    plt.plot(k_range, slope_list, 'k.', label='Data')
    plt.plot(k_range, fit, 'b-',label='Fit')
    plt.xlabel('$k$')
    plt.ylabel('$D(1+k-\\tau_s)$')
    plt.legend(loc=0)
    plt.show()
        