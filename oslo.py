import numpy as np
import matplotlib.pyplot as plt

class Oslo:
    def __init__(self, L):
        self.slopes = np.zeros(L)
        self.thresholds = np.random.randint(1,3,L)
        self.height = 0
        self.aval_size = 0
    
    def drive(self, i = 0):
        self.slopes[i] += 1
        self.height += 1
    
    def relaxation(self, prob = 0.5):
        self.aval_size = 0
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
    
    def simulate(self, steps, prob = 0.5):
        for i in range(steps):
            self.drive()
            self.relaxation(prob)