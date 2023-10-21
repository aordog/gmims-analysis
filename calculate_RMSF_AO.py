#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:01:48 2019

@author: cvaneck

This routine will determine the RMSF and related parameters,
giving the following input information. One of:
a file with channel frequencies and weights
OR
a file with channel frequencies (assumes equal weights)
OR
Input values for mininum frequency, maximum frequency, and channel width.
(assumes equal weights and all channels present)

The outputs are a list of relavant RMSF properties, and a plot of the RMSF
shape.

A. Ordog modified this version to write out phi and RMSF to text file.
"""

#import sys
import argparse
import numpy as np
from RMutils.util_RM import get_rmsf_planes
from matplotlib import pyplot as plt

C = 2.997924538e8 # Speed of light [m/s]

def main():
    """
    Determines what set of input parameters were defined, reads in file or
    generates frequency array as appropriate, and passes frequency and weight
    arrays to the function that works out the RMSF properties.
    """

    descStr = """
    Calculate and plot RMSF and report main properties, given a supplied
    frequency coverage and optional weights (either as second column of
    frequency file, or as separate file)."""

    parser = argparse.ArgumentParser(description=descStr,
                                 formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("freqFile", metavar="freqFile.dat", nargs='?',default=None,
                        help="ASCII file containing frequencies and optionally weights.")
    parser.add_argument("weightFile", metavar="weightFile.dat", nargs='?',
                        help="Optional ASCII file containing weights.")
    
    parser.add_argument("-o", dest="RMSFfile", default='test.txt',
                        help="Name of output RMSF file")
    
    parser.add_argument("-f", dest=("freq_parms"),nargs=3,default=None,
                        help="Generate frequencies (in Hz): minfreq, maxfreq, channel_width",
                        )
    parser.add_argument("-m", dest="phiMax_radm2", type=float, default=None,
                        help="absolute max Faraday depth sampled [Auto, ~10xFWHM].")
    parser.add_argument("-d", dest="dphi_radm2", type=float, default=None,
                        help="Delta phi [Auto, ~10/FWHM].")
    parser.add_argument("-s", dest="plotfile", default=None,
                        help="Filename to save plot to. [do not save]")
    parser.add_argument("-n", dest="plotname", default=None,
                        help="Name of plot [\"Simulated RMSF\"]")
    parser.add_argument("-r", "--super-resolution", action="store_true",
                        help="Optimise the resolution of the RMSF (as per Rudnick & Cotton). "
                        )
    args = parser.parse_args()


    #Check that at least one frequency input has been given:
    if args.freqFile == None and args.freq_parms == None:
        print("Please supply either a file with frequency values or use the -f flag.")
        raise(Exception("No frequency input! Use -h flag for help on inputs."))

    # if args.phiMax_radm2 != None:
    #     if args.phiMax_radm2

    #Order of priority: frequency file takes precedence over -i flag.
    #                   weight file takes precedence over 2nd column of frequency file.
    if args.freqFile != None:
        data=np.genfromtxt(args.freqFile,encoding=None,dtype=None)
        if len(data.shape) == 2:
            freq_array=data[:,0]
            weights_array=data[:,1]
        else:
            freq_array=data
            weights_array=np.ones_like(freq_array)
    else:
        #Generate frequency and weight arrays from intput values.
        freq_array=np.arange(float(args.freq_parms[0]),float(args.freq_parms[1]),
                             float(args.freq_parms[2]))
        weights_array=np.ones_like(freq_array)


    if args.weightFile != None:
        weights_array=np.genfromtxt(args.weightFile,encoding=None,dtype=None)
        if len(weights_array) != len(freq_array):
            raise Exception('Weights file does not have same number of channels as frequency source')


    determine_RMSF_parameters(freq_array,weights_array,args.phiMax_radm2,args.dphi_radm2,args.RMSFfile,args.plotfile,args.plotname, args.super_resolution)

def determine_RMSF_parameters(freq_array,weights_array,phi_max,dphi,RMSF_filename,plotfile=None,plotname=None, super_resolution=False):
    """
    Characterizes an RMSF given the supplied frequency and weight arrays.
    Prints the results to terminal and produces a plot.
    Inputs:
        freq_array: array of frequency values (in Hz)
        weights_array: array of channel weights (arbitrary units)
        phi_max (float): maximum Faraday depth to compute RMSF out to.
        dphi (float): step size in Faraday depth
        plotfile (str): file name and path to save RMSF plot.
        plotname (str): title of plot
    """
    lambda2_array=C**2/freq_array**2
    l2_min=np.min(lambda2_array)
    l2_max=np.max(lambda2_array)
    dl2=np.median(np.abs(np.diff(lambda2_array)))

    if phi_max == None:
        phi_max = 10*2*np.sqrt(3.0) / (l2_max-l2_min)  #~10*FWHM
    if dphi == None:
        dphi = 0.1*2*np.sqrt(3.0) / (l2_max-l2_min)  #~10*FWHM

    phi_array=np.arange(-1*phi_max/2,phi_max/2+1e-6,dphi) #division by two accounts for how RMSF is always twice as wide as FDF.

    RMSFcube, phi2Arr, fwhmRMSFArr, statArr=get_rmsf_planes(
        lambda2_array,
        phi_array,
        weightArr=weights_array,
        fitRMSF=True,
        fitRMSFreal=super_resolution,
        lam0Sq_m2=0 if super_resolution else None,
    )


    #Output key results to terminal:
    #print('RMSF PROPERTIES:')
    #print('Theoretical (unweighted) FWHM:       {:.4g} rad m^-2'.format(3.8 / (l2_max-l2_min)))
    #print('Measured FWHM:                       {:.4g} rad m^-2'.format(fwhmRMSFArr))
    #print('Theoretical largest FD scale probed: {:.4g} rad m^-2'.format(np.pi/l2_min))
    #print('Theoretical maximum FD*:             {:.4g} rad m^-2'.format(np.sqrt(3.0)/dl2))
    #print('*50% bandwdith depolarization threshold, for median channel width in Delta-lambda^2')
    #print('* may not be reliable over very large fractional bandwidths or in data with ')
    #print('differing channel widths or many frequency gaps.')
    #Explanation for below: This code find the local maxima in the positive half of the RMSF,
    #finds the highest amplitude one, and calls that the first sidelobe.
    try:
        x=np.diff(np.sign(np.diff(np.abs(RMSFcube[RMSFcube.size//2:])))) #-2=local max, +2=local min
        y=1+np.where(x==-2)[0]  #indices of peaks, +1 is because of offset from double differencing
        peaks=np.abs(RMSFcube[RMSFcube.size//2:])[y]
    #    print('First sidelobe FD and amplitude:     {:.4g} rad m^-2'.format(phi2Arr[phi2Arr.size//2:][y[np.argmax(peaks)]]))
    #    print('                                     {:.4g} % of peak'.format(np.max(peaks)*100))
    except:
        pass

    
    # Writing out RMSF file
    data_out = np.column_stack((phi2Arr, abs(RMSFcube)))
    np.savetxt(RMSF_filename+'.txt', data_out, delimiter=' ', fmt='%f')
    
    # Writing out parameters file
    params    = ['FWHM ','phi_broad ','phi_max ']
    paramvals = [str(np.round(3.8/(l2_max-l2_min),2)), str(np.round(np.pi/l2_min,2)), str(np.round(np.sqrt(3.0)/dl2,2))]
    with open(RMSF_filename+'_params.txt', 'w') as filehandle:
        for i in range(0,len(params)):
            filehandle.write(f"{params[i]+paramvals[i]}\n")



if __name__ == "__main__":
    main()