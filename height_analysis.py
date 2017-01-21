import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from oslo import Oslo

sys_sizes = [16,32,64,128]
sys_sizes = [8,16,32,64,128]
#sys_sizes = [8,16,32,64]
#sys_sizes = [16,32,64]

#TASK 2a
def plot_height_raw():
    height_sys = []
    for size in sys_sizes:
        h_list = []
        sys = Oslo(size)
        for i in range(sys_sizes[-1]**2):
            sys.simulate(1)
            h_list.append(sys.height)
        height_sys.append(h_list)
        print 'System of size', size, 'complete.'
        
    plt.figure()
    for h in height_sys:
        plt.plot(range(1,sys_sizes[-1]**2+1), h)
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


def plot_height_collapsed():
    height_sys = []
    W= 100
    exp1 = -1
    exp2 = 2
    for size in sys_sizes:
        h_list = []
        sys = Oslo(size)
        for i in range(int(size*size*3)):
            sys.simulate(1)
            h_list.append(sys.height)
        h_list = moving_average(h_list, W)
        height_sys.append((size**exp1)*h_list)
        print 'Size', size,'completed (max', sys_sizes[-1],').'
        
    plt.figure()
    for i in range(len(height_sys)):
        scaled_time = np.arange(2*W+1,int(sys_sizes[i]*sys_sizes[i]*3) +1)
        scaled_time = scaled_time/float(sys_sizes[i]**exp2)
        plt.plot(scaled_time, height_sys[i], label=sys_sizes[i])
    plt.legend(loc=4)
    plt.show()


#TASK 2c
def mean_std_height(L, time):
    h_list = []
    sys = Oslo(L)
    sys.simulate(L*L)
    for i in range(time):
        sys.simulate(1)
        h_list.append(sys.height)
    return np.mean(h_list), np.std(h_list)

def scaling(L, omega_1, a_0, a_1):
    return a_0 + a_1*(L)**(-omega_1)

def plot_height_scaling():
    scaled_means = []
    scaled_std = []
    for size in sys_sizes:
        h_mean, h_std = mean_std_height(size, 10000)
        scaled_means.append(h_mean/float(size))
        scaled_std.append(h_std/float(size))
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
    
    plt.figure(2)
    plt.plot(sys_sizes, scaled_std, '.')
    param_std = curve_fit(scaling, sys_sizes, scaled_std)[0]
    print param_std
    size_range = np.linspace(1, sys_sizes[-1], 100)
    fit_std = scaling(size_range, param_std[0],param_std[1],param_std[2])
    plt.plot(size_range, fit_std, label = 'Fit')
    plt.legend()
    plt.xlabel('System Size')
    plt.ylabel('Scaled Mean Height Standard Dev. $\sigma_h$/L')
    
    plt.show()

    
        