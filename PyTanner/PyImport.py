import pandas as pd
#pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed

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
    
        ranget = r1[(r1.t > mint) & (r1.t < maxt)].t.loc[:1000]
        rangeE = r1[(r1.t > mint) & (r1.t < maxt)].E.loc[:1000]
        ranget = r1.t[:100]
        rangeE = r1.E[:100]
        
        slope = np.average(np.diff(rangeE)/np.diff(ranget)) * 1000
        #print(slope)
        if debug == True:
            print("%.1f mV/s" % slope)
            plt.figure()
            plt.title(path)
            plt.plot(ranget, rangeE)
            plt.plot(r1.t[:100], r1.E[:100])
            plt.show()
        return r1, slope
    else:
        return r1
    
def readtxtCA(path):
    text = pd.read_csv(path,delimiter='\t', encoding='unicode escape')
    r1 = text.rename(columns={"time/s":"t", "Ewe/V": "E", "<I>/mA": "I", "cycle number": "cycle"})
    return r1

# Parent function for readtxtCV function
def importtxtCV(paths, debug=False, filt="", nfilt=""):
    # kwargs: debug=False, filt=CV, nfilt=OCV
    # returns data, ramp rate
    pathlist=[]
    if filt != "":
        if nfilt != "":
            results = Parallel(n_jobs=-1, max_nbytes=1e6, prefer="threads")(delayed(readtxtCV)(path, debug) for path in tqdm(paths, miniters=1, maxinterval=1) if (filt in path)&(nfilt not in path))
            pathlist = [path for path in paths if (filt in path)&(nfilt not in path)]
        else:
            results = Parallel(n_jobs=-1, max_nbytes=1e6, prefer="threads")(delayed(readtxtCV)(path, debug) for path in tqdm(paths, miniters=1, maxinterval=1) if filt in path)
            pathlist = [path for path in paths if (filt in path)]
    elif nfilt != "":
        results = Parallel(n_jobs=-1, max_nbytes=1e6, prefer="threads")(delayed(readtxtCV)(path, debug) for path in tqdm(paths, miniters=1, maxinterval=1) if nfilt not in path)
        pathlist = [path for path in paths if (nfilt not in path)]
    else:
        results = Parallel(n_jobs=-1, max_nbytes=1e6, prefer="threads")(delayed(readtxtCV)(path, debug) for path in tqdm(paths, miniters=1, maxinterval=1))
        pathlist = [path for path in paths]
    r, ramp = [],[]
    #print(results)
    r, ramp = zip(*results)
    return r, ramp, pathlist

# Parent function for readtxtCA function
def importtxtCA(paths, filt="", nfilt=""):
    pathlist=[]
    if filt != "":
        if nfilt != "":
            results = Parallel(n_jobs=-1, max_nbytes=1e6, prefer="threads")(delayed(readtxtCA)(path) for path in tqdm(paths, miniters=1, maxinterval=1) if (nfilt not in path) & (filt in path))
            pathlist = [path for path in paths if (filt in path)&(nfilt not in path)]
        else:
            results = Parallel(n_jobs=-1, max_nbytes=1e6, prefer="threads")(delayed(readtxtCA)(path) for path in tqdm(paths, miniters=1, maxinterval=1) if (filt in path))
            pathlist = [path for path in paths if (filt in path)]
    elif nfilt != "":
        results = Parallel(n_jobs=-1, max_nbytes=1e6, prefer="threads")(delayed(readtxtCA)(path) for path in tqdm(paths, miniters=1, maxinterval=1) if (nfilt not in path))
        pathlist = [path for path in paths if (nfilt not in path)]
    else:
        results = Parallel(n_jobs=-1, max_nbytes=1e6, prefer="threads")(delayed(readtxtCA)(path) for path in tqdm(paths, miniters=1, maxinterval=1))
        pathlist = [path for path in paths]
    return results, pathlist


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


def readtxtPEIS(path):
    text = pd.read_csv(path,delimiter='\t',encoding='unicode escape')
    r1 = text.rename(columns={"freq/Hz":"freq", "Re(Z)/Ohm": "real", "-Im(Z)/Ohm": "imag", "|Z|/Ohm": "comp", "Phase(Z)/deg": "phase","time/s":"t", 
    'cyle number':'cycle'})
    return r1

def importtxtPEIS(paths, filter=""):
    results = Parallel(n_jobs=1, max_nbytes=1e6, prefer="threads")(delayed(readtxtPEIS)(path) for path in tqdm(paths) if filter in path)
    names = [path.split('/')[3] for path in paths if filter in path]
    df = pd.DataFrame({'data':results, 'names':names})
    for index, row in df.iterrows():
        eval = row.data.empty
        if eval == True:
            df.drop(index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df
def filterlist(dfs, fnames, filter=""):
    files = [dfs, fnames]
    print()
    newdata = [data for data, filenames in zip(dfs, fnames) if filter in filenames]
    newnames = [filenames for data, filenames in zip(dfs, fnames) if filter in filenames]
    return newdata, newnames