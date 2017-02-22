import height_analysis as h
import aval_analysis as s
import bonus as bonus
    
    
print 'All of the figures used in the report can be reproduced using this module.'
print 'However, some of the figures in the report took a relatively long time'
print 'to produce, so this module reproduces them for fewer syste sizes and for'
print 'a smaller number of iterations, to allow you to run them relatively quickly.'
print ''
print 'If you wish to run the original results, you can simply add the system'
print 'sizes to the lists in this module. The iterations are set to N=10^6 by'
print 'default, as the 10^8 used took whole days to run and save.'
print ''
print ''
print 'The following is a list of figure numbers and what they correspond to:'
print '1 - implementation consistency test (Figure 2).'
print '2 - loglog plot of heights with time (Figure 3).'
print '3 - scalings of critical time and steady state height. (Figures 4a, 4b).'
print '4 - heights data collapse (Figure 5).'
print '5 - corrections to scaling of steady state height, a0 optimisation'
print '    algorithm, and stand. dev scaling. (Figures 6b,6a,7).'
print '6 - heights probability distributions (Figure 8a).'
print '7 - heights probability distributions collapsed (Figure 8b).'
print '8 - raw and logbinned avalanche size probability data (Figure 9b).'
print '9 - avalanche size probabilities (Figure 10).'
print '10 - avalanche size probabilities collapsed (Figure 11).'
print '11 - moments scaling with system size (Figure 12a).'
print '12 - corrections to scaling of moments scaling law (Figure 12b).'
print '13 - D(1+k-tau) vs k plot, scaling relation (Figure 13).'
print '14 - drop size probability distributions (Figure 14a).'
print '15 - drop size probability distributions collapsed (Figure 14b).'
fig_no = input('Enter the figure number you want to reproduce: ')

h.sys_sizes = [8,16,32,64,128,256]
s.sys_sizes = [8,16,32,64,128,256]
bonus.sys_sizes = [8,16,32,64,128,256]
if fig_no == 1:
    h.plot_BTW()
elif fig_no == 2:
    h.plot_height_raw()
elif fig_no == 3:
    h.plot_crossover_values(128)
elif fig_no == 4:
    h.plot_height_collapsed()
elif fig_no == 5:
    h.plot_height_scaling()
elif fig_no == 6:
    h.plot_height_prob()
elif fig_no == 7:
    h.plot_height_prob_collapsed_alt()
elif fig_no == 8:
    s.sys_sizes = [256]
    s.plot_aval_prob(bin_type='both', task='3a')
elif fig_no == 9:
    s.plot_aval_prob(bin_type='log', task='3b')
elif fig_no == 10:
    s.plot_aval_prob_collapsed()
elif fig_no == 11:
    print 'This WILL take a long time, as the data is not being saved.'
    s.plot_moments()
elif fig_no == 12:
    s.moment_size_scaling(2)
elif fig_no == 13:
    print 'This WILL take a long time, as the data is not being saved.'
    s.moment_analysis()
elif fig_no == 14:
    bonus.plot_drop_prob()
elif fig_no == 15:
    bonus.plot_drop_prob_collapsed()
    

#h.plot_BTW()
#h.plot_height_raw()
#h.plot_crossover_values(128)
#h.plot_height_collapsed()
#h.plot_height_scaling()
#h.plot_height_prob()
#h.plot_height_prob_collapsed_alt()
# The next two go together.
#s.sys_sizes = [256]
#s.plot_aval_prob(bin_type='both', task='3a')
#s.plot_aval_prob(bin_type='log', task='3b')
#s.plot_aval_prob_collapsed()
#s.plot_moments()
#s.moment_size_scaling(2)
#s.moment_analysis()
#bonus.plot_drop_prob()
#bonus.plot_drop_prob_collapsed()