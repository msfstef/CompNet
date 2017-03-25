import kdistanalysis as kd

print 'All of the figures used in the report can be reproduced using this module.'
print 'However, some of the figures in the report took a relatively long time'
print 'to produce, so this module reproduces them for smaller network sizes and for'
print 'a smaller number of iterations, to allow you to run them relatively quickly.'
print ''
print 'The following is a list of figure numbers and what they correspond to:'
print 'Preferential Attachment:'
print '1 - multiple m for given N (Figure 2a).'
print '2 - one m one N logbin and linear (Figure 2b).'
print '3 - KS statistics + figure. (Figure 3).'
print '4 - chi squared statistic figure (Figure 4).'
print '5 - one m multiple N (Figure 5a)'
print '6 - collapse of multiple N (Figure 5b).'
print '7 - kmax vs N (Figure 6a).'
print '8 - kmax vs N factor for many m (Figure 6b).'
print 'Random Attachment:'
print '9 - multiple m for given N (Figure 7a).'
print '10 - one m one N logbin and linear (Figure 7b).'
print '11 - KS statistics + chisquare.'
print '12 - one m multiple N (Figure 8a)'
print '13 - collapse of multiple N (Figure 8b).'
print '14 - kmax vs N (Figure 9a).'
print '15 - kmax vs N factor for many m (Figure 9b).'
print 'Random Walk Attachment:'
print '16 - Random Walk for m=2, regular (Figure 10a and 10b).'
print '17 - Random Walk for m=1, bipartite (Figure 11a and 11b).'
fig_no = input('Enter the figure number you want to reproduce: ')

kd.method = 0
if fig_no == 1:
    kd.plot_dist_multiple_m(5,1e5,100)
elif fig_no == 2:
    kd.plot_dist_raw_log(1,1e5,100)
elif fig_no == 3:
    print 'Percent of null hypothesis not rejected: ', kd.two_sample_test(1,1e5,100,100)
    kd.ks_test_dist(1,1e5,100)
elif fig_no == 4:
    kd.plot_chi_squared(1,1e5,100,200,500)
elif fig_no == 5:
    kd.plot_dist_multiple_N(1,1e5,100)
elif fig_no == 6:
    kd.plot_dist_collapsed(m=1, N_max=1e5, runs=100)
elif fig_no == 7:
    print 'Exponent with error: ',kd.fit_k_max_vs_N(5,1e5)[0]
elif fig_no == 8:
    kd.find_prop_k_max(5, 1e5)

kd.method = 1
if fig_no == 9:
    kd.plot_dist_multiple_m(5,1e5,100)
elif fig_no == 10:
    kd.plot_dist_raw_log(1,1e5,100)
elif fig_no == 11:
    print 'Percent of null hypothesis not rejected: ', kd.two_sample_test(1,1e5,100,100)
    kd.plt.figure(0)    
    kd.ks_test_dist(1,1e5,100)
    kd.plt.figure(1)
    kd.plot_chi_squared(3,1e5,100,10,30)
elif fig_no == 12:
    kd.plot_dist_multiple_N(3,1e5,100)
elif fig_no == 13:
    kd.plot_dist_collapsed(m=3, N_max=1e5, runs=100)
elif fig_no == 14:
    kd.fit_k_max_vs_N(5,1e5)
elif fig_no == 15:
    kd.find_prop_k_max(5, 1e5)

kd.method = 2
if fig_no == 16:
    kd.plot_k_dist_walker(2,1e5,100,5, process='logbin')
elif fig_no == 17:
    kd.plot_k_dist_walker(1,1e5,100,5, process='logbin')
