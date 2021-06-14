import os
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
from scipy.stats import norm
from scipy.stats import uniform
from astropy.io import fits
from astropy import units as u
import pandas as pd
from emcee.utils import MPIPool
import pickle
import mpi4py
#from mpi4py import MPI
import sys
#sys.modules["mpi4py"] = None
#!pip install --user lenstools
import lenstools
from lenstools import ConvergenceMap
from IPython.display import Image

mpi4py.rc.recv_mprobe = False
mpi4py.rc.threads=False
ng = 1024  # number of grids
map_side_deg = 5*u.degree

theta_g2 = 2./np.sqrt(2)
theta = 5.

pix = theta/ng 
tin = theta_g2/60./pix
offset = int(2*ceil(tin))
print('starting..')

def map_stats(pair_lp_run): # here it will read a LP folder and its run folder from where I will get kappa23
    lpi, runi = pair_lp_run
    if lpi < 10:
        lpi2 = '00'+str(lpi)
    elif 10 <= lpi < 100:
        lpi2 = '0'+str(lpi)
    elif lpi==100:
        lpi2 = str(lpi)
        
    if runi < 10:
        runi2 = '00'+str(runi)
    elif 10 <= runi < 100:
        runi2 = '0'+str(runi)
    elif runi==100:
        runi2 = str(runi)
         
    
    fname2 = '/global/cscratch1/sd/jialiu/kappaTNG/kappaTNG-Hydro/LP'+str(lpi2)+'/run'+str(runi2)+'/kappa23.dat'
    with open(fname2, 'rb') as f:
        dummy = np.fromfile(f, dtype="int32", count=1)
        kappa3 = np.fromfile(f, dtype="float", count = ng*ng)
        dummy = np.fromfile(f, dtype="int32", count=1)
        
        kappa2 = kappa3.reshape((ng, ng))
            
        conv_map = ConvergenceMap(data=kappa2, angle=map_side_deg)
        smoothed_conv_map1= conv_map.smooth(theta_g2*u.arcmin, kind='gaussianFFT')
        
        new_map = smoothed_conv_map1.data[offset:1024-offset,offset:1024-offset]

        smoothed_conv_map = ConvergenceMap(data=new_map, angle=map_side_deg)
        
        kappa_bin_edges = np.arange(-0.24875, 0.99875+0.0025,0.0025)
        
        # PDF    
        pdf_test = smoothed_conv_map.pdf(kappa_bin_edges)
            
        #Peaks
        peaks_test = smoothed_conv_map.peakCount(kappa_bin_edges)
            
        #Minima
        minima_test = ConvergenceMap(data=-smoothed_conv_map.data,angle=map_side_deg).peakCount(kappa_bin_edges)
            
        #print(minima_test)
    fname3 = '/global/cscratch1/sd/jialiu/kappaTNG/kappaTNG-Dark/LP'+str(lpi2)+'/run'+str(runi2)+'/kappa23.dat'
    
    with open(fname3, 'rb') as f1:
        dummy1 = np.fromfile(f1, dtype="int32", count=1)
        kappa4 = np.fromfile(f1, dtype="float", count = ng*ng)
        dummy1 = np.fromfile(f1, dtype="int32", count=1)
        
        kappa5 = kappa4.reshape((ng, ng))
        conv_map2 = ConvergenceMap(data=kappa5, angle=map_side_deg)
        smoothed_conv_map3= conv_map2.smooth(theta_g2*u.arcmin, kind='gaussianFFT')
        
        new_map2 = smoothed_conv_map3.data[offset:1024-offset,offset:1024-offset]

        smoothed_conv_map2 = ConvergenceMap(data=new_map2, angle=map_side_deg)
            
        # PDF
        dmo_pdf_test = smoothed_conv_map2.pdf(kappa_bin_edges)
            
        #Peaks
        dmo_peaks_test = smoothed_conv_map2.peakCount(kappa_bin_edges)
        #Minima
        
        dmo_minima_test = ConvergenceMap(data=-smoothed_conv_map2.data,angle=map_side_deg).peakCount(kappa_bin_edges)
            
    
    return [peaks_test[0][51:101],peaks_test[1][51:101],dmo_peaks_test[1][51:101]]
    
    
lp_run=[[lpi, runi] for lpi in range(1,101) for runi in range(1,101)]
print('loading MPI..')
pool=MPIPool()

if not pool.is_master():
    pool.wait()
    sys.exit(0)
    
print('running')

out=pool.map(map_stats, lp_run)

pool.close()
print ('DONE-DONE-DONE')


save('smpeak_offset_2', out)


sys.exit(0)
