import numpy as np
import ctypes

# Importing the C function for relaxation.
relaxlib = ctypes.cdll.LoadLibrary('relax.so')

class Oslo:
    def __init__(self, L):
        """
        Initialises system for the given size L.
        """
        self.size = L
        self.slopes = np.zeros(L, int)
        self.thresholds = np.random.randint(1,3,L)
        self.height = 0
        self.aval_size = 0
        self.drop_size = 0
    
    def drive(self, i = 0):
        """
        Adds grain to position i, resets counters for avalanche size, drop size.
        """
        self.slopes[i] += 1
        self.height += 1
        self.aval_size = 0
        self.drop_size = 0
    
    def relaxation(self, prob = 0.5):
        """
        prob - float, from 0 to 1, determines probability that threshold slope
            for a given site is set to 1 after relaxation, otherwise set
            to 2. (Default : 0.5)
        
        Checks the system for any sites with slopes above their corresponding
        threshold slope and relaxes them. Continues until no site is above
        its threshold.
        """
        diff = self.thresholds-self.slopes
        relax_ind = np.where(diff < 0)[0]
        relax_len = len(relax_ind)
        while relax_len != 0:
            self.aval_size += relax_len
            for i in relax_ind:
                if i == 0:
                    self.slopes[i] -= 2
                    self.slopes[i+1] += 1
                    self.height -= 1
            
                elif i == len(self.slopes)-1:
                    self.slopes[i] -= 1
                    self.slopes[i-1] += 1
                    self.drop_size += 1
                else:
                    self.slopes[i] -= 2
                    self.slopes[i-1] += 1
                    self.slopes[i+1] += 1
                    
                self.thresholds[i] = np.random.choice([1,2],p=[prob,1.-prob])
                
            diff = self.thresholds-self.slopes
            relax_ind = np.where(diff < 0)[0]
            relax_len = len(relax_ind)
    
    def relaxation_fast(self, prob=0.5):
        """
        See relaxation method above.
        
        Uses compiled C code with different algorithm to relax the system
        more efficiently and allow for larger system sizes to be used.
        """
        hsd = np.array([self.height, self.aval_size, self.drop_size], dtype=int)
        relaxlib.restype = None
        relaxlib.relax(ctypes.c_int(self.size), 
                       ctypes.c_double(prob), 
                        np.ctypeslib.as_ctypes(hsd), 
                        np.ctypeslib.as_ctypes(self.slopes), 
                        np.ctypeslib.as_ctypes(self.thresholds))
        self.height = hsd[0]
        self.aval_size = hsd[1]
        self.drop_size = hsd[2]
    
    def simulate(self, steps, prob = 0.5):
        """
        steps - int, number of grains to add to the system.
        prob - float, from 0 to 1, determines probability that threshold slope
            for a given site is set to 1 after relaxation, otherwise set
            to 2. (Default : 0.5)
        
        Adds grain to the system, and relaxes it completely, and repeats the
        process for the given number of steps.
        """
        for i in xrange(steps):
            self.drive()
            self.relaxation_fast(prob)
