'''
################################################################################
# James Clough 2015
# jrc309@ic.ac.uk
# 2016 Complexity & Networks course
#
# log_bin.py v1.3
# 14/01/2016
#
# Usage instructions for log binning:
# 
# to import the log_bin functions:
# from log_bin import *
# 
# to use the log_bin function:
# arguments:
# data            - list - the numbers we want to get a probability distribution of
# bin_start       - float - the left edge of the first bin
# first_bin_width - float - the width of the first bin
# a               - float - growth factor - each bin is a times larger than the bin to its left
# datatype        - string - either 'float' or 'integer' so we know how to normalise bin widths
#
# returns:
# centres - list - the centre of each bin (in log space - ie. geometric mean)
# counts  - list - the probability of data being in each 
#
#
# Examples - available below
#
# - test_float() - Generates some power-law distributed real numbers and log-
#                  bins them. The blue points are the original numbers - the red
#                  line is the log-binning - we can see how useful it is because
#                  we get a few more orders of mangnitude of data
#
# - test_int()   - As above, but with integers
#
# - test_too_low_a() - Shows a plot where the growth factor, a, is too low
#                      and there is a kink in the line due to there being too
#                      few points in the bin
#
# Notes:
#    
# - The data will be binned with the width of bins growing exponentially so that
# even when there is less than one expected point per value we can still 
# estimate the probability distribution.
#
# - It matters whether your data are real numbers, or integers as this will change
# how we should normalise the height of each bin - use the 'datatype' argument
# to tell the function whether you are using integers or real numbers
# This is not asking whether the data you have provided are Python integers, or floats
# but rather whether you want to treat the data as integers or floats for the 
# purpose of the log binning.
#
# - Make sure to use a sensible value for 'a', which is how much larger each bin
# is than the bin to its left. If a is too small then you will have too many
# bins and too few points per bin. If it is too large then there won't be
# many bins.
#
# - If you want to have negative values on the x-axis then something will break
# when finding the centre of the bin that contains 0 - if you need to do this
# you will need a workaround as log-binning negative data is not well defined.
# 
# - The solution implemented here is not the most efficient one possible.
# possible. Feel free to tinker with this code to make it better.
#
# - There may be some bugs here - please email me if you find one.
################################################################################
'''

################################################################################
import numpy as np
from numba import jit, int_, float_
################################################################################
# BINNING FUNCTIONS
################################################################################
# frequency
#
# arguments:
# data - list of values
#
# returns:
# values - list of values
# frequency - list of frequencies of values
################################################################################
def frequency(data):
    from collections import Counter
    c = Counter(data)
    return np.array([float(x) for x in c.keys()]), np.array([float(x) for x in c.values()])
    
################################################################################
# lin_bin
# linear binning of data
#
# arguments:
# data     - list - the numbers we want to get the probability distribution of
# num_bins - integer - the number of bins to use
#
# returns:
# bin_centres   - list - the centres of each bin (arithmetic mean)
# normed_counts - list - the probability of data being in each bin
################################################################################
def lin_bin(data, num_bins):
    data = np.array(data)
    num_datapoints = float(len(data))
    counts, bin_edges = np.histogram(data, num_bins)
    bin_centres = (bin_edges[1:] + bin_edges[:-1])/2
    normed_counts = counts/num_datapoints
    return bin_centres, normed_counts

################################################################################
# log_bin
# log bin data of either integers, or real numbers
#
# arguments:
# data            - list - the numbers we want to get a probability distribution of
# bin_start       - float - the left edge of the first bin
# first_bin_width - float - the width of the first bin
# a               - float - growth factor - each bin is a times larger than the bin to its left
# datatype        - string - either 'float' or 'integer' so we know how to normalise bin widths
# 

#s      - boolean - if True, zeros are not counted in the normalisation of the probability distribution
#
# returns:
# centres - list - the centre of each bin (in log space - ie. geometric mean)
# counts  - list - the probability of data being in each bin
################################################################################
@jit
def log_bin(data, bin_start=1., first_bin_width=1., a=2., datatype='int', drop_zeros=True):
    # ensure data is numpy array of floats
    if drop_zeros:
        data = np.array(data)[data!=0]
    else:
        data = np.array(data)
    
    max_x = float(np.max(data))
    a = float(a)
    first_bin_width = float(first_bin_width)
    
    
    max_power = np.ceil(np.log(1+((a-1)*max_x)/first_bin_width)/np.log(a) -1.)
    
    widths = first_bin_width*np.power(a,np.arange(max_power+1, dtype='float'))
    
    bins = np.cumsum(np.concatenate([np.array([bin_start]), widths]))
    
    # find how many datapoints are in each bin
    # counts[i] is how many points are there in the bin whose left edge is bins[i]
    indices = np.digitize(data, bins[1:])
    counts = np.bincount(indices)/float(data.size)
    bins = bins[:len(counts)+1]
    widths = widths[:len(counts)]
    

    # we want to give a 'centre' of the bin to plot its height from
    # if the data is approximately power law distributed then the most
    # representative would be the geometric mean, since in it is in the middle
    # in poutsoktonos log space
    bin_indices = np.arange(len(bins)-1) 
    bins = np.array(bins)
    if datatype == 'float':
        centres = np.sqrt(np.roll(bins, -1)* bins)[:-1]
    elif datatype == 'int':
        centres = np.empty(len(bin_indices))
        widths = np.diff(np.ceil(bins),1)
        for i in bin_indices:
            centres[i] = geometric_mean(np.arange(np.ceil(bins[i]), np.ceil(bins[i+1])))
    counts /= widths
    return centres, counts

# returns the geometric mean of a list
@jit(float_[:](int_[:]))
def geometric_mean(x):
    s = len(x)
    y = np.log(x)
    z = np.sum(y)/s
    return np.exp(z)

