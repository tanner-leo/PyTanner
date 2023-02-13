import pandas as pd
#pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
from dataclasses import dataclass, field
from PyTanner import Plots as p
import eclabfiles as ecf

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
def folder2files(folder, filter=[''], remover=['']):
    # Folder should be like "./txt/"
    scan = os.scandir(folder)
    names = []
    if type(filter) == str:
        filter = [filter]
    if type(remover) == str:
        remover = [remover]
    if filter == '':
        names = [("%s%s" % (folder, scan.name)) for scan in scan]
    else:
        namesfil = [("%s" % (scan.name)) for scan in scan]
        for filter in filter:
            namesfil = [name for name in namesfil if filter in name]
        if remover != ['']:
            for nfilter in remover:
                namesfil = [name for name in namesfil if nfilter not in name]
        names = [("%s%s" % (folder, name)) for name in namesfil]
        
    return names

def process(files, diagnose=False):
    # datalist = []
    # metalist = []
    appender = []
    for file in files:
        if diagnose == True:
            print(file)
        data, meta = ecf.process(file)
        # datalist.append(data)
        # metalist.append(meta)
        appender.append({'data':pd.DataFrame.from_dict(data),'meta':meta,'technique':meta['settings']['technique'],'reference_electrode':meta['settings']['reference_electrode']})
    df = pd.DataFrame.from_dict(appender)
    return df


def readtxtPEIS(path):
    text = pd.read_csv(path,delimiter='\t',encoding='unicode escape')
    r1 = text.rename(columns={"freq/Hz":"freq", "Re(Z)/Ohm": "real", "-Im(Z)/Ohm": "imag", "|Z|/Ohm": "comp", "Phase(Z)/deg": "phase","time/s":"t", 
    'cyle number':'cycle', "time/s":"t", "Ewe/V": "E", "<I>/mA": "I", "cycle number": "cycle", "<Ewe>/V": "E","control/mA":"icontrl"})
    return r1

def importtxtPEIS(paths, filter="",namelevel=3):
    results = Parallel(n_jobs=1, max_nbytes=1e6, prefer="threads")(delayed(readtxtPEIS)(path) for path in tqdm(paths) if filter in path)
    names = [path.split('/')[namelevel] for path in paths if filter in path]
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

@dataclass
class ECdata:
    df: pd.DataFrame = field(repr=False)
    comment: str = field(repr=True, default="Blank Description")
    types: list[str] = field(repr=True, default="None")

    def __post_init__(self):
        self.df['type'], self.df['obj'] = np.nan, np.nan
        for index, row in self.df.iterrows():
            name = row.names
            name2 = row.data.columns
            if "PEIS_C" in name:
                row.type = "PEIS"
                row.obj = p.PEIS(row.data, name=row.names)
            elif "CP_C" in name:
                row.type = "CP"
                row.obj = p.CP(row.data, name=row.names)
            elif "CA_C" in name:
                row.type = "CA"
                row.obj = p.CA(row.data, name=row.names)
            elif "_CV_C" in name:
                row.type = "CV"
            elif "OCV_C" in name:
                row.type = "OCV"
                row.obj = p.OCV(row.data, name=row.names)
            else:
                if "freq" in name2:
                    row.type = "PEIS"
                    row.obj = p.PEIS(row.data, name=row.names)
                elif "Icontrl" in name2:
                    row.type = "CP"
                    row.obj = p.CP(row.data, name=row.names)
                elif "Econtrl" in name2:
                    row.type = "CA"
                    row.obj = p.CA(row.data, name=row.names)
                elif "CV_C" in name2:
                    row.type = "CV"
                elif "OCV_C" in row.names2:
                    row.type = "OCV"
                    row.obj = p.OCV(row.data, name=row.names)
            self.df.iloc[index] = row
        self.types = list(self.df.type)
        self.df['comment'] = np.nan

    def CP(self): return self.df[self.df.type == "CP"]
    def CA(self): return self.df[self.df.type == "CA"]
    def CV(self): return self.df[self.df.type == "CV"]
    def PEIS(self): return self.df[self.df.type == "PEIS"]
    def OCV(self): return self.df[self.df.type == "OCV"]