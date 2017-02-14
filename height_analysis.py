import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from oslo import Oslo
font = {'family' : 'Arial',
        'size'   : 16}
matplotlib.rc('font', **font)
cm = plt.get_cmap('nipy_spectral')

sys_sizes = [16,32,64,128]
sys_sizes = [8,16,32,64,128,256,512,1024,2048]
#sys_sizes = [8,16,32,64,128,256,512]




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
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/9) for i in range(9)])
    for i in range(len(sys_sizes)):
        ax.plot(range(1,int(sys_sizes[-1]**2.+1)), height_sys[i], 
                label='L = ' + str(sys_sizes[i]))
    
    plt.xlabel('Time (# of grains added)')
    plt.ylabel('Pile Height h')
    plt.xlim(0,sys_sizes[-1]*sys_sizes[-1])
    plt.legend(loc=2, ncol=2)
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


def plot_crossover_values(L_max=128):
    t_data = []
    h_data = []
    size_range = np.arange(4,int(L_max)+1)
    for size in size_range:
        data_point = find_crossover(size)
        t_data.append(data_point[0])
        h_data.append(data_point[1])
        print 'Size', size,'/',size_range[-1],'completed.'
      
    plt.figure(1)
    plt.plot(size_range, t_data, '.', label='Data')
    time_fit_coeff = np.polyfit(size_range, t_data, 2)
    time_fit = (time_fit_coeff[0]*size_range*size_range)  
                #+time_fit_coeff[1]*size_range + time_fit_coeff[2])
    print "Time Coeff:",time_fit_coeff
    plt.plot(size_range, time_fit, label='$t_c \propto L^2$ Fit', lw=2.5)
    plt.xlabel('System Size L')
    plt.ylabel('Critical Time $t_c$')
    plt.xlim(4,int(L_max))
    plt.legend(loc=2)
    

    plt.figure(2)
    plt.plot(size_range, h_data, '.', label='Data')
    h_fit_coeff = np.polyfit(size_range, h_data, 1)
    h_fit = h_fit_coeff[0]*size_range + h_fit_coeff[1]
    print "Height Coeff:",h_fit_coeff
    plt.plot(size_range, h_fit, label='$h \propto L$ Fit', lw=2.5)
    plt.xlabel('System Size L')
    plt.ylabel('Steady State Height $h$')
    plt.xlim(4,int(L_max))
    plt.legend(loc=2)
    
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
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/9) for i in range(9)])
    for i in xrange(len(height_sys)):
        scaled_time = np.arange(2*W+1,int(sys_sizes[i]*sys_sizes[i] + 1e5) +1)
        scaled_time = scaled_time/float(sys_sizes[i]**float(exp2))
        ax.loglog(scaled_time, height_sys[i], 
                  label='L = '+ str(sys_sizes[i]), lw=2)
        if i == len(height_sys)-1:
            L = sys_sizes[i]
            param, cov = np.polyfit(np.log10(scaled_time)[int(0.2*L*L):int(0.6*L*L)], 
                               np.log10(height_sys[i])[int(0.2*L*L):int(0.6*L*L)], 1,
                               cov = True)
            print 'Slope of transient:', param[0]
            print 'Covariance matrix:', cov
    plt.xlabel('Scaled Time $t/L^2$')
    plt.ylabel('Scaled Height $h/L$')
    plt.legend(loc=4, ncol=2)    
    
    plt.show()


#TASK 2c
def gen_height_list(L, time=1e8, gen=False, save=True):
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

def plot_height_scaling(a_0_guess = 1.733):
    scaled_means = []
    scaled_std = []
    for size in sys_sizes:
        h_mean, h_std = mean_std_height(size)
        scaled_means.append(h_mean/float(size))
        scaled_std.append(h_std)#/float(size**0.26))
    
    a_0_list, error_list = [], []
    step_dir = 1
    step = 0.001
    old_error = 0.1
    error = 0.01
    a_0 = a_0_guess
    iter_no = 0
    while step > 1e-6:
        if error > old_error:
            step_dir = -1*step_dir
            step = step/2.
        a_0 += step_dir*step
        scaled_means_estimate = np.divide(scaled_means, float(a_0))
        scaled_means_estimate = 1. - scaled_means_estimate
        param, cov_height = np.polyfit(np.log10(sys_sizes), 
                            np.log10(scaled_means_estimate), 1, cov=True)
        old_error = np.copy(error)
        error = np.sqrt(np.abs(cov_height[0][0]))
        a_0_list.append(a_0)
        error_list.append(error)
        iter_no += 1
    
    
    fig = plt.figure(3) 
    ax1 = fig.add_subplot(111)
    ax1.plot(np.arange(iter_no) +1, a_0_list, 'b-', label='$a_0$')
    ax1.set_xlabel('Number of Iterations')
    ax1.set_ylabel('Estimate of $a_0$', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlim(1,iter_no)
    ax2 = ax1.twinx()
    ax2.plot(np.arange(iter_no) +1, error_list, 'r-', label='$\omega_1$ error')
    ax2.set_xlabel('Number of Iterations')
    ax2.set_ylabel('Error on $\omega_1$', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_xlim(1,iter_no)
    
    print 'Iterations to optimal a_0 value:', iter_no    
    print "a_0 value:", a_0
    print "omega_1 value:", -param[0]
    print 'Error in omega_1:', error
    plt.figure(1)
    plt.loglog(sys_sizes, scaled_means_estimate, '.', label='Data')
    size_range = np.linspace(sys_sizes[0], sys_sizes[-1], 100)
    fit_means = param[0]*np.log10(size_range) + param[1]
    fit_means = np.power(10., fit_means)
    plt.loglog(size_range, fit_means, label = 'Fit')
    plt.legend(loc=0)
    plt.xlabel('System Size L')
    plt.ylabel('Scaled Mean Height  $1-\overline{h}/a_0L$')
    plt.xlim(sys_sizes[0], sys_sizes[-1]*1.1)
    plt.ylim(np.min(scaled_means_estimate)*0.8,
             np.max(scaled_means_estimate)*1.2)
    plt.show()
    
    plt.figure(2)
    plt.loglog(sys_sizes,scaled_std, '.', label = 'Data')
    param, cov_std = np.polyfit(np.log10(sys_sizes),np.log10(scaled_std),1, 
                                cov=True)
    print 'Standard Dev. scaling with L:', param[0]
    print 'Error in Stand. Dev.:', np.sqrt(np.abs(cov_std[0][0]))
    fit_std = param[0]*np.log10(size_range) + param[1]
    fit_std = np.power(10., fit_std)
    plt.loglog(size_range, fit_std, label = 'Fit')
    plt.xlabel('System Size L')
    plt.ylabel('Mean Height Standard Dev. $\sigma_h$')
    plt.xlim(sys_sizes[0], sys_sizes[-1]*1.1)
    plt.ylim(np.min(scaled_std)*0.8,
             np.max(scaled_std)*1.2)
    plt.legend(loc=0)
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
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/9) for i in range(9)])
    for i in range(len(sys_sizes)):
        ax.semilogx(range_list[i],prob_dist[i], label="L = "+str(sys_sizes[i]))
    plt.xlabel('Height $h$')
    plt.ylabel('Prob. $P(h;L)$')
    plt.legend(ncol=2, loc = 1)
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
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/9) for i in range(9)])
    for i in xrange(len(sys_sizes)):
        ax.plot(scaled_range[i],scaled_prob[i], lw=1.5, 
                label='L = ' + str(sys_sizes[i]))
    
    plt.xlabel('Scaled Height $(h - \overline{h})/\sigma_h$')
    plt.ylabel('Scaled Prob. $\sigma_h P(h;L)$')
    plt.legend()
    plt.show()

#plot_height_prob_collapsed()
#plot_height_scaling()
