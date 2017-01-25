import numpy as np
import ctypes

# Importing the C function for relaxation.
relaxlib = ctypes.cdll.LoadLibrary('relax.so')

class Oslo:
    def __init__(self, L):
        self.size = L
        self.slopes = np.zeros(L, int)
        self.thresholds = np.random.randint(1,3,L)
        self.height = 0
        self.aval_size = 0
    
    def drive(self, i = 0):
        self.slopes[i] += 1
        self.height += 1
        self.aval_size = 0
    
    def relaxation(self, prob = 0.5):
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
                else:
                    self.slopes[i] -= 2
                    self.slopes[i-1] += 1
                    self.slopes[i+1] += 1
                    
                self.thresholds[i] = np.random.choice([1,2],p=[prob,1.-prob])
                
            diff = self.thresholds-self.slopes
            relax_ind = np.where(diff < 0)[0]
            relax_len = len(relax_ind)
    
    def relaxation_alt(self, prob=0.5):
        hs = np.array([self.height, self.aval_size])
        relaxlib.restype = None
        relaxlib.relax(ctypes.c_int(self.size), 
                       ctypes.c_float(prob), 
                        np.ctypeslib.as_ctypes(hs), 
                        np.ctypeslib.as_ctypes(self.slopes), 
                        np.ctypeslib.as_ctypes(self.thresholds))
        self.height = hs[0]
        self.aval_size = hs[1]

    
    def simulate(self, steps, prob = 0.5):
        for i in xrange(steps):
            self.drive()
            self.relaxation_alt(prob)
