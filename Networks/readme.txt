COMPLEXITY PROJECT README
===========================
There should be 14 files:
- gen_network.exe		~ compiled network generation module
- gen_network.cpp		~ C++ network generation module
- simplegraph.cpp		~ networks modules provided by Tim Evans
- simplegraph.h			~ required to compile gen_network.cpp
- log_bin.py			~ log binning function for smoothing data
- kdistanalysis.py		~ utility functions to analyse results
- gen_data.bat			~ tool to generate required data to reproduce results
- 'data' folder			~ required for gen_network.exe to save data
- RESULTS.py			~ USE THIS TO REPRODUCE RESULTS
------
- README.txt
- MousafeirisS-NetworksReport.pdf
===========================

To generate required data, run the gen_data.bat and let it run.

To reproduce the results from the report, the RESULTS.py should be run
and simple instructions will be printed that will ask you to input a
number corresponding to which figure you want to reproduce.

To use it, execute the module in your IDE or run the following in a 
terminal within the same directory as the files:

> python RESULTS.py

The program will prompt you to insert a number, from 1 to 17, that
corresponds to a figure in the report, and it will produce that result.
However, some of the figures in the report took a relatively long time
to produce, so this module reproduces them for smaller network sizes and for
a smaller number of iterations, to allow you to run them relatively quickly.

=======================================================================

The following is a list of figure numbers and what they correspond to:
Preferential Attachment:
1 - multiple m for given N (Figure 2a).
2 - one m one N logbin and linear (Figure 2b).
3 - KS statistics + figure. (Figure 3).
4 - chi squared statistic figure (Figure 4).
5 - one m multiple N (Figure 5a)
6 - collapse of multiple N (Figure 5b).
7 - kmax vs N (Figure 6a).
8 - kmax vs N factor for many m (Figure 6b).
Random Attachment:
9 - multiple m for given N (Figure 7a).
10 - one m one N logbin and linear (Figure 7b).
11 - KS statistics + chisquare.
12 - one m multiple N (Figure 8a)
13 - collapse of multiple N (Figure 8b).
14 - kmax vs N (Figure 9a).
15 - kmax vs N factor for many m (Figure 9b).
Random Walk Attachment:
16 - Random Walk for m=2, regular (Figure 10a and 10b).
17 - Random Walk for m=1, bipartite (Figure 11a and 11b).