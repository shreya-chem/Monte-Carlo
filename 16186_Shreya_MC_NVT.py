# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:47:36 2020

@author: SV
"""
import numpy as np
import time
import sys

"""
========================================================================
==========================Defining Parameters===========================
WE ARE WORKING WITH REDUCED UNITS
========================================================================
"""

nc=int(input("Enter the number of unit cells for number of particles (Use 3) : " ))
ncycle=int(input("Enter the maximum number of cycles (Use 100000) : " ))
cell=1.0/nc; cell2 = cell/2
n=4*(nc**3)

temp = 1.0          # temperature
rcut = 2.5          # cut-off distance
rmin = 0.75         # minimum distance for overlap
rho = 0.6           # density
drmax = 0.15        # maximum r
boxl = (n/rho)**(1.0/3)     # length of actual box
rcutboxl = rcut/boxl        # cut-off distance in box-units
rcutboxlsq = rcutboxl*rcutboxl  # square cut-off in box-units
rsq_ovlp = rmin*rmin            # square minimum overlap distance
print(boxl)                     
moves = 0                       # no. of moves accepted
ntrial = 0                      # no. of total trials
moveratio = 0.0                 # accepted move ratio
delta = 0.0                     # dU/temperature
r = (np.zeros(3*n, dtype=float)).reshape(3,n)   # array to store coordinates of all particles
ri = np.zeros(n, dtype=float)       # array to store coordintes of particle being trnslated

np.random.seed(3)  #seed to always generate same set of random nos.

"""
========================================================================
============================Initial Positions===========================
Gives initial coordinates of FCC lattice within [0,1]
========================================================================
"""

def initial_pos(): 
    r[0][1]=cell2; r[1][1]=cell2; r[2][1]=0.0
    r[0][2]=0.0; r[1][2]=cell2; r[2][2]=cell2
    r[0][3]=cell2; r[1][3]=0.0; r[2][3]=cell2  
    m=0                                                                                            #generating initial coordinates in fcc lattice
    for Iz in range(0,nc):
        for Iy in range(0,nc):
            for Ix in range(0,nc):
                for Iref in range(0,4):
                    r[0][Iref+m]=r[0][Iref]+(cell*Ix)
                    r[1][Iref+m]=r[1][Iref]+(cell*Iy)
                    r[2][Iref+m]=r[2][Iref]+(cell*Iz)
                m=m+4   
    return()
initial_pos()
"""
=========================================================================
=============================L.J. Potential==============================
Takes - initial coordinates 
Returns - initial energy using Lennard Jones potential, overlap value
=========================================================================
"""

def LJ_pot(R):
    Epot=0.0
    overlap = False                      
    if (rcutboxl<0.5):
        for i in range(0,n-1):
            for j in range(i+1,n):
                rxij=R[0,i]-R[0,j]; rxij=rxij-round(rxij)                                                                                                  #pbc-x
                ryij=R[1,i]-R[1,j]; ryij=ryij-round(ryij)                                                                                                    #pbc-y
                rzij=R[2,i]-R[2,j]; rzij=rzij-round(rzij)                                                                                                     #pbc-z
                rsq=(rxij*rxij)+(ryij*ryij)+(rzij*rzij)
                if (rsq < rcutboxlsq):
                    rsq=rsq*boxl*boxl
                    if(rsq<=rsq_ovlp):
                        overlap = True
                    else:
                        rsqcubeinv=1/(rsq*rsq*rsq)
                        Epot=Epot+4.0*rsqcubeinv*(rsqcubeinv-1.0)
                        overlap = False 
    return(Epot,overlap)
EPn=np.zeros((ncycle+1), dtype=float)  # array to store energies at each MC cycle
EPn[0],overlap_initial = LJ_pot(r)        

"""
========================================================================
=============Calculate PE of atom i with all other atoms================
Takes - Coordinates of i th atom and its index 'i'
Returns - Energy of atom 'i' with all other atoms 'j', overlap value
========================================================================
"""

def LJpot_ofonewithallother(rith,i):
    partial_pot=0.0
    over = False
    if (rcutboxl<0.5):
        for j in range(n):
            if(j!=i):
                drx=rith[0]-r[0,j]; drx=drx-round(drx)                                                                                                      #pbc-x
                dry=rith[1]-r[1,j]; dry=dry-round(dry)                                                                                                     #pbc-y
                drz=rith[2]-r[2,j]; drz=drz-round(drz)                                                                                                    #pbc-z
                rsq=(drx**2)+(dry**2)+(drz**2)
                if (rsq < rcutboxlsq):
                    rsq=rsq*boxl*boxl
                    if((1/rsq) > (1/rsq_ovlp)):
                        over = True
                    r6inv=1/(rsq*rsq*rsq)
                    partial_pot=partial_pot+4.0*r6inv*(r6inv-1.0)
                    over = False        
    return(partial_pot,over)


"""
=========================================================================
========================Random Translate Vector==========================
Takes - drmax, old vector to be translated
Returns - translated vector with MIC and PBC
=========================================================================
"""

def ran_trans_vec(drmax,old):
    rtv = np.zeros(3)
    zeta = np.zeros(3)
    zeta[0] = np.random.uniform(-1,1)
    zeta[1] = np.random.uniform(-1,1)
    zeta[2] = np.random.uniform(-1,1)
    rtv = old + (zeta*(drmax/boxl))
    rtv = rtv%1
    return(rtv)
    
"""
========================================================================
===============================Metropolis===============================
Takes - delta value
Returns - accept value which implies whether the move is accepted or not
========================================================================
"""

def metropolis(delta):
    accept = True
    zeta = 0.0
    exp_guard = 75.0
    if (delta>exp_guard):
        accept = False
    elif (delta<0.0):
        accept = True
    else:
        zeta = np.random.uniform(0,1)
        accept = np.exp(-1.0*delta)>zeta
    return(accept)

"""
=========================================================================
=================Store required coordinates in Array=====================
Takes - coordinates to be stored, counter variables m, t
Gives - arrays RX, RY, RZ after every prt_tag cycles to be written in file
=========================================================================
"""

prt_tag=int(input("Enter the number of steps after which you want the runtime data to be printed (Use 100 or any factor of ncycle) : " ))
sv_tag=int(input("Enter the number of steps after which you want the coordinates to be written in file (Use 100 or any factor of ncycle) : " ))
q=ncycle/sv_tag

RX=np.zeros(int(n*(q+1)), dtype=float)
RY=np.zeros(int(n*(q+1)), dtype=float)
RZ=np.zeros(int(n*(q+1)), dtype=float)
cyclenumber=np.zeros(len(RX)) 
m=1

for i in range(n):
    RX[i]=r[0,i]*boxl
    RY[i]=r[1,i]*boxl
    RZ[i]=r[2,i]*boxl

def store(Rs,m,t):
    for k in range(n):
        RX[k+(n*m)]=r[0,k]*boxl
        RY[k+(n*m)]=r[1,k]*boxl
        RZ[k+(n*m)]=r[2,k]*boxl
        cyclenumber[k+(n*m)]=int(t)
    return()
    
"""
=========================================================================
==============================MC Simulation==============================
MC cycle operation
=========================================================================
"""

potn=0.0
acpot=0.0
acpot_sq=0.0
energy=EPn[0]
EPn[0]=EPn[0]/n
print("Initial Energy :",EPn[0])
l=0
trials=0

start = time.clock()
if((overlap_initial==False)and(rcutboxl<0.5)):
    for stp in range(ncycle):
        moves=0
        for i in range(n):
            pot_old, overlap1 = LJpot_ofonewithallother(r[:,i],i)
            if(overlap1==False):
                ri = ran_trans_vec(drmax,r[:,i]) 
                pot_new, overlap2 = LJpot_ofonewithallother(ri,i)
                if(overlap2==False):
                    delta = (pot_new-pot_old)/temp
                    if(metropolis(delta)):
                        energy = energy + delta
                        r[:,i] = ri
                        moves=moves+1
                trials=trials+1     
            else:
                sys.exit("OVERLAP")
            ntrial=ntrial+1     
            potn = energy/n
        EPn[stp+1] = potn
        moveratio=moves/n
        if(moveratio>0.55):
            drmax = drmax*1.05
        elif(moveratio<0.45):
            drmax = drmax*0.95
            
        if((stp+1)%sv_tag==0):
            store(r,m+l,stp+1)
            l=l+1
        if((stp+1)%prt_tag==0):
            print("Cycle no.: ",stp+1,"No. of performed trials: ",trials, "Move ratio: ",moveratio, "EP per particle : ",potn)
else:
    print("Sorry, your values do not satisfy the conditions of this NVT Monte Carlo Simulation")

end=time.clock()
print('Time taken to execute the code is ',end-start,' seconds')
"""
========================================================================
==========================Data to be collected==========================
Saving coordinates and energies in separate files
Calculating fluctuation in energies
========================================================================
"""

item=np.zeros((ncycle+1), dtype=float)
for i in range(ncycle+1):
    item[i]=i
   
coordinates=(np.vstack((cyclenumber,RX,RY,RZ))).T
np.savetxt('mc_coordinates.dat', coordinates, delimiter='\t', header='Cycle Number, X, Y, Z')

en=(np.vstack((item,EPn))).T
np.savetxt('mc_energy.dat', en, delimiter='\t', header='Potential Energies at each MC Cycle')

print("Final system configuration:")
print((r*boxl).T)

fluct=0.0; var=0.0
acpot = np.sum(EPn)
pot_avg = np.mean(EPn)
acpot_sq = np.sum(EPn**2)
var = np.mean(EPn*EPn)-(pot_avg*pot_avg)
if(var>0.0):
    fluct = np.sqrt(var)

print("Average Per Particle Potential Energy of the system: ",pot_avg)
print("Average Squared Per Particle Potential Energy of the system: ",np.mean(EPn**2))
print("Fluctuation (SD) in the Per Particle Potential Energy of the system: ",fluct)


"""
========================================================================
================================END=====================================
========================================================================
"""        
