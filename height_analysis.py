import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from oslo import Oslo

sys_sizes = [16,32,64,128]
sys_sizes = [8,16,32,64,128,256,512]

#TASK 2a
def plot_height_raw():
    height_sys = []
    for size in sys_sizes:
        h_list = []
        sys = Oslo(size)
        for i in range(int(sys_sizes[-1]**2.)):
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
    print time_fit_coeff
    plt.plot(size_range, time_fit)
    
    plt.figure(2)
    plt.plot(size_range, h_data, '.')
    h_fit_coeff = np.polyfit(size_range, h_data, 1)
    h_fit = h_fit_coeff[0]*size_range + h_fit_coeff[1]
    print h_fit_coeff
    plt.plot(size_range, h_fit)
    
    plt.show()

#TASK 2b
def moving_average(arr, W):
    N = 2*W+1
    ma = np.cumsum(arr, dtype=float)
    ma[N:] = ma[N:] - ma[:-N]
    return ma[N - 1:] / N


def plot_height_collapsed(exp1 = -1, exp2 = 2, W = 100):
    height_sys = []
    #W= 100
    #exp1 = -1
    #exp2 = 2
    for size in sys_sizes:
        h_list = []
        sys = Oslo(size)
        for i in range(int(size*size*3)):
            sys.simulate(1)
            h_list.append(sys.height)
        h_list = moving_average(h_list, W)
        height_sys.append((size**float(exp1))*h_list)
        print 'Size', size,'completed (max', sys_sizes[-1],').'
        
    plt.figure()
    for i in range(len(height_sys)):
        scaled_time = np.arange(2*W+1,int(sys_sizes[i]*sys_sizes[i]*3) +1)
        scaled_time = scaled_time/float(sys_sizes[i]**float(exp2))
        plt.plot(scaled_time, height_sys[i], label=sys_sizes[i])
    plt.legend(loc=4)
    plt.show()


#TASK 2c
def gen_height_list(L, time=1e6, gen=True):
    h_list = np.empty(int(time))
    if not gen:
        return np.load('hlist'+str(L)+'.npy')
    sys = Oslo(L)
    sys.simulate(L*L)
    
    for i in range(int(time)):
        sys.simulate(1)
        h_list[i] = sys.height
    
    np.save('hlist'+str(L)+'.npy',np.array(h_list))
    return h_list

def mean_std_height(L):
    h_list = gen_height_list(L)
    return np.mean(h_list), np.std(h_list)

def scaling(L, omega_1, a_0, a_1):
    return a_0 + a_1*(L)**float(-omega_1)


def plot_height_scaling():
    scaled_means = []
    scaled_std = []
    for size in sys_sizes:
        h_mean, h_std = mean_std_height(size)
        scaled_means.append(h_mean/float(size))
        scaled_std.append(h_std)#/float(size**0.26))
        print 'Size', size,'completed (max', sys_sizes[-1],').'
    
    plt.figure(1)
    plt.plot(sys_sizes, scaled_means, '.')
    param = curve_fit(scaling, sys_sizes, scaled_means)[0]
    print param
    size_range = np.linspace(1, sys_sizes[-1], 100)
    fit_means = scaling(size_range, param[0],param[1],param[2])
    plt.plot(size_range, fit_means, label = 'Fit')
    plt.legend(loc=4)
    plt.xlabel('System Size')
    plt.ylabel('Scaled Mean Height h/L')
    
    plt.show()
    
    plt.figure(2)
    plt.loglog(sys_sizes,scaled_std)
    print np.polyfit(np.log(sys_sizes),np.log(scaled_std),1)[0]
    plt.show()    
    
    
    #plt.figure(3)
    #plt.plot(sys_sizes, scaled_std, '.')
    #param_std = curve_fit(scaling, sys_sizes, scaled_std)[0]
    #print param_std
    #size_range = np.linspace(1, sys_sizes[-1], 100)
    #fit_std = scaling(size_range, param_std[0],param_std[1],param_std[2])
    #plt.plot(size_range, fit_std, label = 'Fit')
    #plt.legend()
    #plt.xlabel('System Size')
    #plt.ylabel('Scaled Mean Height Standard Dev. $\sigma_h$/L')
    
    #plt.show()


#TASK 2d
def gen_height_prob(L, time=1e6):
    h_list = gen_height_list(L,time)
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
        print 'Size', size,'completed (max', sys_sizes[-1],').'
    
    for i in range(len(sys_sizes)):
        plt.plot(range_list[i],prob_dist[i], label=sys_sizes[i])
    plt.legend()
    plt.show()

def plot_height_prob_collapsed():
    exp1 = 0.221
    exp2 = 1
    a_0 = 1.728
    prob_dist, range_list = [], []
    for size in sys_sizes:
        h_prob, h_range = gen_height_prob(size)
        prob_dist.append(h_prob)
        range_list.append(h_range)
        print 'Size', size,'completed (max', sys_sizes[-1],').'
    
    for i in range(len(sys_sizes)):
        scaled_prob = np.multiply(sys_sizes[i]**float(exp1) ,prob_dist[i])
        scaled_range = np.divide(range_list[i]-a_0*sys_sizes[i]**float(exp2), 
                                 float(sys_sizes[i]**exp1))
        
        plt.plot(scaled_range,scaled_prob, label=sys_sizes[i])
    plt.legend()
    plt.show()

#plot_height_prob_collapsed()
#plot_height_scaling()