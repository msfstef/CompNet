COMPLEXITY PROJECT README
===========================
There should be 14 files:
- oslo.py				~ Oslo model algorithm
- relax.c				~ relaxation algorithm in C
- relax.so				~ compiled relaxation algorithm
- height_analysis.py	~ utility functions for height analysis
- aval_analysis.py		~ utility functions for avalanche size analysis
- bonus.py				~ utility functions for drop size analysis
- log_bin.py			~ log binning function for smoothing data
- RESULTS.py			~ USE THIS TO REPRODUCE RESULTS
------
DLLs needed to run C code if PATH is not configured correctly
- ntdll.dll
- kernel32.dll
- KernelBase.dll
- cygwin1.dll
------
- README.txt
- MousafeirisS-ComplexityReport.pdf
===========================

To reproduce the results from the report, the RESULTS.py should be run
and simple instructions will be printed that will ask you to input a
number corresponding to which figure you want to reproduce.

The DLL files were included as for some computers where the Python path
was not configured well some of these standard files needed to be in the
same directory to properly run.

To use it, execute the module in your IDE or run the following in a 
terminal within the same directory as the files:

> python RESULTS.py

The program will prompt you to insert a number, from 1 to 15, that
corresponds to a figure in the report, and it will produce that result.
However, some of the figures in the report took a relatively long time
to produce, so this module reproduces them for fewer syste sizes and for
a smaller number of iterations, to allow you to run them relatively quickly.

If you wish to run the original results, you can simply add the system
sizes to the lists in this module. The iterations are set to N=10^6 by
default, as the 10^8 used took whole days to run and save.

~~~~~~~~~~~~~~~~~~~~~~~~
TROUBLESHOOTING:
~~~~~~~~~~~~~~~~~~~~~~~~
- Issue: code produces some kind of memory error when loading.
= Solution: the C wrapper might have issues in your Python setup,
but it definitely runs on college computers. Try changing line 94
of the oslo.py module from relaxation_fast(prob) to relaxation(prob),
which is the slower python version.

- Issue: input/output method from RESULTS.py does not work.
= Solution: the functions used to reproduce the results were also put
in comments at the end of the RESULTS.py module, so you can uncomment
the desired one and run the code. They are in the order given at the
end of this document.

- Issue: some results take too long to run.
= Solution(?): the code was left to run over several days for some
of the results, and data was saved and reused multiple times, taking up
tens of gigabytes of space. It is not possible to reproduce the exact
same results as in the report in a small time scale, but if even the ones 
set to default can't run in reasonable time, try removing system sizes 
from lines 35-37 in the RESULTS.py module.

=======================================================================

The following is a list of figure numbers and what they correspond to:
1 - implementation consistency test (Figure 2).
2 - loglog plot of heights with time (Figure 3).
3 - scalings of critical time and steady state height. (Figures 4a, 4b).
4 - heights data collapse (Figure 5).
5 - corrections to scaling of steady state height, a0 optimisation
    algorithm, and stand. dev scaling. (Figures 6b,6a,7).
6 - heights probability distributions (Figure 8a).
7 - heights probability distributions collapsed (Figure 8b).
8 - raw and logbinned avalanche size probability data (Figure 9b).
9 - avalanche size probabilities (Figure 10).
10 - avalanche size probabilities collapsed (Figure 11).
11 - moments scaling with system size (Figure 12a).
12 - corrections to scaling of moments scaling law (Figure 12b).
13 - D(1+k-tau) vs k plot, scaling relation (Figure 13).
14 - drop size probability distributions (Figure 14a).
15 - drop size probability distributions collapsed (Figure 14b).


P.S. Sorry for the long README, have a nice day!