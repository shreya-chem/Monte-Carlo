# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:14:20 2020
Extracting data from data file and plotting
@author: SV
"""

import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('mc_energy.dat',skiprows=1,delimiter='\t')
cycle_no = data[:,0]
PE = data[:,1]

plt.figure()
plt.plot(cycle_no,PE)
plt.ylabel('Potential Energy per particle in Reduced Units')
plt.xlabel('Number of MC cycles')
plt.title('Monte Carlo Simulation - NVT')
plt.show()

plt.figure()
plt.plot(cycle_no,PE*108)
plt.ylabel('Potential Energy in Reduced Units')
plt.xlabel('Number of MC cycles')
plt.title('Monte Carlo Simulation - NVT')
plt.show()

ex=np.mean(PE)
ex2=np.mean(PE*PE)
print("SD = ", np.sqrt(ex2-ex*ex))