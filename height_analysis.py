import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from oslo import Oslo

sys_sizes = [16,32,64,128]
sys_sizes = [8,16,32,64,128,256,512,1024,2048]

#TASK 2a
def plot_height_raw():
    height_sys = []
    for size in sys_sizes:
        h_list = []
        sys = Oslo(size)
        for i in xrange(int(sys_sizes[-1]**2.)):
            sys.simulate(1)
            h_list.append(sys.height)
        height_sys.append(h_list)
        print 'System of size', size, 'complete.'
        
    plt.figure()
    for h in height_sys:
        plt.plot(range(1,int(sys_sizes[-1]**2.+1)), h)
    plt.show()
    

def find_crossover(L):
    sys = Oslo(L)
    var = np.linspace(-1000,-400,L*L)
    h_list = []
    while np.mean(var[:L*L/2]) < np.mean(var[L*L/2:]) :
        sys.simulate(1)
        var = np.roll(var,-1)
        var[-1] = sys.height
        h_list.append(sys.height)
    h_max = np.mean(var)
    t_crit = np.where(h_list-h_max > 0)[0][0]
    return t_crit, h_max


def plot_crossover_values():
    t_data = []
    h_data = []
    size_range = np.arange(4,64)
    for size in size_range:
        data_point = find_crossover(size)
        t_data.append(data_point[0])
        h_data.append(data_point[1])
        print 'Size', size,'/',size_range[-1],'completed.'
      
    plt.figure(1)
    plt.plot(size_range, t_data, '.')
    time_fit_coeff = np.polyfit(size_range, t_data, 2)
    time_fit = (time_fit_coeff[0]*size_range*size_range +  
                time_fit_coeff[1]*size_range + time_fit_coeff[2])
    print "Time Coeff:",time_fit_coeff
    plt.plot(size_range, time_fit)
    
    plt.figure(2)
    plt.plot(size_range, h_data, '.')
    h_fit_coeff = np.polyfit(size_range, h_data, 1)
    h_fit = h_fit_coeff[0]*size_range + h_fit_coeff[1]
    print "Height Coeff:",h_fit_coeff
    plt.plot(size_range, h_fit)
    
    plt.show()

#TASK 2b
def moving_average(arr, W):
    N = 2*W+1
    ma = np.cumsum(arr, dtype=float)
    ma[N:] = ma[N:] - ma[:-N]
    return ma[N - 1:] / N


def plot_height_collapsed(exp1 = 1, exp2 = 2, W = 100):
    height_sys = []

    for size in sys_sizes:
        h_list = []
        sys = Oslo(size)
        for i in xrange(int(size*size +1e5)):
            sys.simulate(1)
            h_list.append(sys.height)
        h_list = moving_average(h_list, W)
        height_sys.append(h_list/float(size**float(exp1)))
        print 'Size', size,'completed (max', sys_sizes[-1],').'
        
    plt.figure()
    for i in xrange(len(height_sys)):
        scaled_time = np.arange(2*W+1,int(sys_sizes[i]*sys_sizes[i] + 1e5) +1)
        scaled_time = scaled_time/float(sys_sizes[i]**float(exp2))
        plt.loglog(scaled_time, height_sys[i], label=sys_sizes[i])
        if i == len(height_sys)-1:
            L = sys_sizes[i]
            param = np.polyfit(np.log10(scaled_time)[int(0.2*L*L):int(0.6*L*L)], 
                               np.log10(height_sys[i])[int(0.2*L*L):int(0.6*L*L)], 1)
            print 'Slope of transient: ', param[0]
    plt.legend(loc=4)    
    
    plt.show()


#TASK 2c
def gen_height_list(L, time=1e7, gen=False, save=True):
    h_list = np.empty(int(time))
    if not gen:
        print 'Size', L,'completed (max', sys_sizes[-1],').'
        return np.load('hlist'+str(L)+'.npy')
    sys = Oslo(L)
    sys.simulate(L*L)
    
    for i in xrange(int(time)):
        sys.simulate(1)
        h_list[i] = sys.height
    
    if save:
        np.save('hlist'+str(L)+'.npy',np.array(h_list))
    print 'Size', L,'completed (max', sys_sizes[-1],').'
    return h_list

def mean_std_height(L):
    h_list = gen_height_list(L)
    return np.mean(h_list), np.std(h_list)

def plot_height_scaling(a_0 = 1.733):
    scaled_means = []
    scaled_std = []
    for size in sys_sizes:
        h_mean, h_std = mean_std_height(size)
        scaled_means.append(h_mean/float(size))
        scaled_std.append(h_std)#/float(size**0.26))
    
    plt.figure(1)
    scaled_means_estimate = np.divide(scaled_means, float(a_0))
    scaled_means_estimate = 1. - scaled_means_estimate
    plt.loglog(sys_sizes, scaled_means_estimate, '.', label='Data')
    param = np.polyfit(np.log10(sys_sizes), 
                       np.log10(scaled_means_estimate), 1)
    print "a_0 value:", a_0
    print "omega_1 value:", -param[0]
    size_range = np.linspace(sys_sizes[0], sys_sizes[-1], 100)
    fit_means = param[0]*np.log10(size_range) + param[1]
    fit_means = np.power(10., fit_means)
    plt.loglog(size_range, fit_means, label = 'Fit')
    plt.legend(loc=0)
    plt.xlabel('System Size')
    plt.ylabel('Scaled Mean Height h/a_0*L')
    plt.xlim(sys_sizes[0], sys_sizes[-1]*1.1)
    plt.ylim(np.min(scaled_means_estimate)*0.8,
             np.max(scaled_means_estimate)*1.2)
    plt.show()
    
    plt.figure(2)
    plt.loglog(sys_sizes,scaled_std, '.', label = 'Data')
    param = np.polyfit(np.log10(sys_sizes),np.log10(scaled_std),1)
    print 'Standard Dev. scaling with L:', param[0] 
    fit_std = param[0]*np.log10(size_range) + param[1]
    fit_std = np.power(10., fit_std)
    plt.loglog(size_range, fit_std, label = 'Fit')
    plt.xlabel('System Size')
    plt.ylabel('Mean Height Standard Dev.')
    plt.xlim(sys_sizes[0], sys_sizes[-1]*1.1)
    plt.ylim(np.min(scaled_std)*0.8,
             np.max(scaled_std)*1.2)
    plt.show()


#TASK 2d
def gen_height_prob(L):
    h_list = gen_height_list(L)
    time = float(len(h_list))
    h_hist = np.histogram(h_list, np.arange(np.min(h_list),
                                    np.max(h_list)+2,1))
    h_prob = h_hist[0]/float(time)
    h_range = h_hist[1][:-1]
    return h_prob, h_range

def plot_height_prob():
    prob_dist, range_list = [], []
    for size in sys_sizes:
        h_prob, h_range = gen_height_prob(size)
        prob_dist.append(h_prob)
        range_list.append(h_range)
    
    for i in range(len(sys_sizes)):
        plt.plot(range_list[i],prob_dist[i], label=sys_sizes[i])
    plt.legend()
    plt.show()

def plot_height_prob_collapsed():
    exp1 = 0.221
    exp2 = 1.
    a_0 = 1.73
    prob_dist, range_list = [], []
    for size in sys_sizes:
        h_prob, h_range = gen_height_prob(size)
        prob_dist.append(h_prob)
        range_list.append(h_range)
    
    for i in xrange(len(sys_sizes)):
        scaled_prob = np.multiply(sys_sizes[i]**float(exp1) ,prob_dist[i])
        scaled_range = np.divide(range_list[i]-a_0*sys_sizes[i]**float(exp2), 
                                 float(sys_sizes[i]**exp1))
        
        plt.plot(scaled_range,scaled_prob, label=sys_sizes[i])
    plt.legend()
    plt.show()

def plot_height_prob_collapsed_alt():
    scaled_prob, scaled_range = [], []
    for size in sys_sizes:
        h_prob, h_range = gen_height_prob(size)
        h_mean, h_std = mean_std_height(size)
        scaled_prob.append(np.multiply(h_prob,h_std))
        scaled_range.append(np.divide(h_range-h_mean,h_std))
    
    for i in xrange(len(sys_sizes)):
        plt.plot(scaled_range[i],scaled_prob[i], label=sys_sizes[i])
    plt.legend()
    plt.show()

#plot_height_prob_collapsed()
#plot_height_scaling()
