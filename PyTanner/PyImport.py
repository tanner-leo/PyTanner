import pandas as pd
#pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import os

pathlist = []

# importing txt files and determining the scan rates
def readtxtCV(path, debug=False, calc_ramp=True):
    text = pd.read_csv(path,delimiter='\t')
    pathlist.append(path)
    if calc_ramp==True:
        r1 = text.rename(columns={"time/s":"t", "Ewe/V": "E", "<I>/mA": "I", "cycle number": "cycle"})
        maxt = r1[r1.E == max(r1.E)].t.values[0]
        maxE = r1[r1.E == max(r1.E)].E.values[0]
        mint = r1[r1.t == min(r1.t)].t.values[0]
        minE = r1[r1.t == mint].E.values[0]
    
        ranget = r1[(r1.t > mint) & (r1.t < maxt)].t[:1000]
        rangeE = r1[(r1.t > mint) & (r1.t < maxt)].E[:1000]
        ranget = r1.t[:100]
        rangeE = r1.E[:100]
        
        slope = np.average(np.diff(rangeE)/np.diff(ranget)) * 1000
        print("%.1f mV/s" % slope)
    
        if debug == True:
            plt.figure()
            plt.title(path)
            plt.plot(ranget, rangeE)
            plt.plot(r1.t[:100], r1.E[:100])
            plt.show()
        return r1, slope
    else:
        return r1, pathlist
    
def readtxtCA(path):
    print(path)
    text = pd.read_csv(path,delimiter='\t', encoding='unicode escape')
    r1 = text.rename(columns={"time/s":"t", "Ewe/V": "E", "<I>/mA": "I", "cycle number": "cycle"})
    return r1

# Parent function for readtxtCV function
def importtxtCV(paths, debug=False, filt="", nfilt=""):
    # kwargs: debug=False, filt=CV, nfilt=OCV
    # returns data, ramp rate
    
    if filt != "":
        if nfilt != "":
            results, pathlist = [readtxtCV(path, debug) for path in paths if (filt in path)&(nfilt not in path)]
        else:
            results, pathlist = [readtxtCV(path, debug) for path in paths if filt in path]
    elif nfilt != "":
        results, pathlist = [readtxtCV(path, debug) for path in paths if nfilt not in path]
    else:
        results, pathlist = [readtxtCV(path, debug) for path in paths]
    r, ramp = [],[]
    r, ramp = zip(*results)
    return r, ramp, pathlist

# Parent function for readtxtCA function
def importtxtCA(paths, filt="", nfilt=""):
    if filt != "":
        if nfilt != "":
            results = [readtxtCA(path) for path in paths if (nfilt not in path) & (filt in path)]
        else:
            results = [readtxtCA(path) for path in paths if (filt in path)]
    elif nfilt != "":
        results = [readtxtCA(path) for path in paths if (nfilt not in path)]
    else:
        results = [readtxtCA(path) for path in paths]
    return results


# Example of how to use the import txt function
'''
path = ["./txt/GC_WE_sweeps_1p2-p8_001_01_CV_C01.mpt", "./txt/GC_WE_sweeps_1p2-p8_001_02_CV_C01.mpt", "./txt/GC_WE_sweeps_1p2-p8_001_03_CV_C01.mpt", "./txt/GC_WE_sweeps_1p2-p8_001_04_CV_C01.mpt", "./txt/GC_WE_sweeps_1p2-p8_001_05_CV_C01.mpt"]
r1, ramp1 = [], []
debug = False
results = [readtxt(path, debug) for path in path]
r1, ramp1 = zip(*results)
'''
def folder2files(folder):
    # Folder should be like "./txt/"
    scan = os.scandir(folder)
    names = []
    for scan in scan:
        #print(scan)
        names.append("%s%s" % (folder, scan.name))
    return names