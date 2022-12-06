#import pandas as pd
#pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
#import os
import cmath
from dataclasses import dataclass, field
import pandas as pd


def plotCVseries(datalist, parameter, x_axis ="E (V vs Ag/AgCl)", y_axis="i (mA)"):
    for df, param in zip(datalist, parameter):
        plt.plot(df.E, df.I, label="%.3f mV/s" % parameter)
    plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()

class Impedance:
    def __init__(self, Y, freq):
        self.imag = Y.imag
        self.real = Y.real
        self.comp = Y
        self.freq = freq
        self.phase = np.angle(Y, deg=True)
        
    def dual_plots(self, ylim=[0,1000], fw=10, fname='./Figure.png'):
        Y = self.comp
        freq = self.freq
        fig, ax = plt.subplots(1,2)
        fig.set_figwidth(fw)
        ax[0].plot(Y.real, abs(Y.imag), '-bo', markersize=2.5)
        ax[0].set_ylabel("|$Z_{imag}$| (Ohms)")
        ax[0].set_xlabel("|$Z_{real}$| (Ohms)")
        ax[0].set_aspect('equal', adjustable='datalim')
        ax[1].plot(freq, abs(Y))
        ax[1].set_yscale('log')
        ax[1].set_xscale('log')
        ax[0].set_ylim(ylim)
        ax2 = ax[1].twinx()
        phase = np.angle(Y, deg=True)
        ax2.plot(freq, phase, 'r')
        ax2.set_ylabel("Phase Angle (deg)")
        ax[1].set_ylabel("|$Z$| (Ohms)")
        ax[1].set_xlabel("Frequency (Hz)")
        plt.tight_layout()
        size = fig.gca()
        fig.savefig(fname, format='png', dpi=300)
    
    def bode(self, fname='./Figure_bode.png'):
        fig, ax = plt.subplots(1,1)
        ax.plot(self.freq, abs(self.comp), 'black')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax2 = ax.twinx()
        ax2.plot(self.freq, self.phase, 'red')
        ax2.set_ylabel("Phase Angle (deg)")
        ax.set_ylabel("|$Z$| (Ohms)")
        ax.set_xlabel("Frequency (Hz)")
        ax2.spines['right'].set_color('red')
        ax2.spines['left'].set_color('black')
        fig.savefig(fname, format='png', dpi=300)
    
    def nyquist(self, fname='./Figure_nyquist.png'):
        fig, ax = plt.subplots(1,1)
        ax.plot(self.real, abs(self.imag))
        ax.set_ylabel("|$Z_{imag}$| (Ohms)")
        ax.set_xlabel("|$Z_{real}$| (Ohms)")
        ax.set_aspect('equal', adjustable='datalim')
        #ax.set_ylim(ylim)
        fig.savefig(fname, format='png', dpi=300)

def plt_bode(self, fname='./Figure_bode.png', save=False, title=''):
    fig, ax = plt.subplots(1,1)
    ax.plot(self.freq, abs(self.comp), 'black')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax2 = ax.twinx()
    ax2.plot(self.freq, self.phase, 'red')
    ax2.set_ylabel("Phase Angle (deg)")
    ax.set_ylabel("|$Z$| (Ohms)")
    ax.set_xlabel("Frequency (Hz)")
    ax2.spines['right'].set_color('red')
    ax2.spines['left'].set_color('black')
    if save == True:
        plt.savefig(fname, format='png', dpi=300)
    if title != "":
        plt.title(title)
    
def plt_dual_plots(self, ylim=[0,1000], fw=10, fname='./Figure.png', save=False):
        Y = self.comp
        freq = self.freq
        fig, ax = plt.subplots(1,2)
        fig.set_figwidth(fw)
        ax[0].plot(Y.real, abs(Y.imag), '-bo', markersize=2.5)
        ax[0].set_ylabel("|$Z_{imag}$| (Ohms)")
        ax[0].set_xlabel("|$Z_{real}$| (Ohms)")
        ax[0].set_aspect('equal', adjustable='datalim')
        ax[1].plot(freq, abs(Y))
        ax[1].set_yscale('log')
        ax[1].set_xscale('log')
        ax[0].set_ylim(ylim)
        ax2 = ax[1].twinx()
        phase = np.angle(Y, deg=True)
        ax2.plot(freq, phase, 'r')
        ax2.set_ylabel("Phase Angle (deg)")
        ax[1].set_ylabel("|$Z$| (Ohms)")
        ax[1].set_xlabel("Frequency (Hz)")
        plt.tight_layout()
        size = fig.gca()
        if save == True:
            fig.savefig(fname, format='png', dpi=300)
    


def plt_nyquist(df, save=False, fname="./Figure_nyquist.png", title=""):
    plt.plot(df.real, df.imag, 'ro', markersize=3)
    plt.axis('square')
    plt.xlim(-1,max(df.real)+5)
    plt.ylim(-1,max(df.real)+5)
    plt.xlabel("$Z_{real}$ (ohms)")
    plt.ylabel("$Z_{imag}$ (ohms)")
    plt.tight_layout()
    if title != "":
        plt.title(title)
    plt.tight_layout()
    if save == True:
        plt.savefig(fname, format='png', dpi=300, bbox_inches='tight')
    plt.show()


@dataclass
class PEIS:
    data: pd.DataFrame = field(repr=False)
    name: str = field(repr=True, default="PEIS Dataframe")
    description: str = field(repr=False, default="description here")
    #real: pd.Series 
    #imag: pd.Series
    
    def __post_init__(self):
        self.real = self.data.real
        self.imag = self.data.imag
        self.complex = self.data.comp
        self.freq = self.data.freq

    def nyquist(self, save=False, fname="Fig.png", title=""):
        plt_nyquist(self, save=save, fname=fname, title=title)

    def bode(self, save=False, fname='./Fig.png', title=''):
        plt_bode(self.data, save=save, fname=fname, title=title)
    
    def dual_plots(self, ylim=[0,1000], fw=10, fname='./Figure.png', save=False):
        plt_dual_plots(self.data, ylim=ylim, fw=fw, fname=fname, save=save)


    def write_description(self, descrp):
        self.description = descrp


    

@dataclass
class PEISs:
    data: pd.DataFrame = field(repr=False)
    slicer: str
    units: str = ""
    xax: str = ""
    yax: str = ""
    description: str = field(default_factory=str)
    def __post_init__(self):
        self.sl_val = self.data[self.slicer]
        self.datas = self.data.values
        self.labels = self.data[self.slicer]+ " " + self.units
        self.databackup = self.data
        if 'cycle' in self.data:
            if max(self.data.cycle) > 1:
                print("Multiple Cycles Detected")
        
    def nyquist(self, save=False, fname='./Fig.png'):
        for index, row in self.data.iterrows():
            label = row[self.slicer] + " " + self.units
            data = row.data
            #print(data)
            plt.plot(data.real, data.imag, label="%s" % (label))
        plt.xlabel(self.xax)
        plt.ylabel(self.yax)
        plt.legend()
        if save == True:
            plt.savefig(fname, format='png', dpi=300)
        plt.show()

    

            
    def add_units(self, unit):
        self.units = unit
        self.labels = self.data[self.slicer] + " " + self.units
    
    def add_axes(self, xaxis, yaxis):
        self.xax = xaxis
        self.yax = yaxis

    def remove(self, items=list[int]):
        #print(type(items))
        if type(items)==type(1):
            try:
                self.data.drop(index=items, inplace=True)
                self.data = self.data.reset_index(drop=True)
            except:
                print("already removed")
        else:
            for item in items:
                try:
                    self.data.drop(index=item, inplace=True)
                    self.data = self.data.reset_index(drop=True)
                except:
                    print("already removed")
        
    
    def reset(self):
        self.data = self.databackup


@dataclass
class CP:
    data: pd.DataFrame
    name: str = field(repr=True, default="CP")
    description: str = field(repr=False, default="description")
    timeunits: str = 's'
    Eunits: str = "V"
    reference: str = field(repr=True, default="Ag/AgCl")

    def __post_init__(self):
        self.data = self.data.rename(columns={"freq/Hz":"freq", "Re(Z)/Ohm": "real", "-Im(Z)/Ohm": "imag", "|Z|/Ohm": "comp", "Phase(Z)/deg": "phase","time/s":"t", 
    'cyle number':'cycle', "time/s":"t", "Ewe/V": "E", "<I>/mA": "I", "cycle number": "cycle", "<Ewe>/V": "E","control/mA":"Icontrl"})
        self.columns = self.data.columns
        self.cycles = self.data.cycle.unique()

    def column(self):
        list(self.data.columns)

    def change_units(self, timeunits='s', Eunits='V'):
        if (self.timeunits == timeunits) & (self.Eunits == Eunits):
            return
        if timeunits == 'm':
            if self.timeunits == 's':
                self.data.t = self.data.t / 60
                self.timeunits = 'm'
        else:
            if self.timeunits == 'm':
                self.data.t = self.data.t * 60
                self.timeunits = 's'     
        if Eunits == 'mV':
            self.data.E = self.data.E *1000
            self.data.Econtrl = self.data.Econtrl * 1000


    def plot(self, timeunit='s', save=False, figpass = "", cycles=False, fname='./Fig.png'):
        if figpass == "":
            fig = plt.figure()
        else:
            fig = plt.figure(figpass)
        if cycles == False:
            plt.plot(self.data.t, self.data.E)
            plt.xlabel("Time (%s)" % self.timeunits)
            plt.ylabel("E (%s vs %s)" % (self.Eunits, self.reference))
            
        else:
            for cycle in self.cycles:
                d = self.data[self.data.cycle == cycle]
                plt.plot(d.t, d.E, marker='.', linestyle="None")
                
            plt.xlabel("Time (%s)" % self.timeunits)
            plt.ylabel("I (%s)" % self.Eunits)
        if save == True:
            plt.savefig(fname, format='png', dpi=300)
        plt.show()

        return fig

@dataclass
class CA:
    data: pd.DataFrame
    name: str = field(repr=True, default="CA")
    description: str = field(repr=False, default="description")
    timeunits: str = 's'
    Iunits: str = "mA"
    reference: str = field(repr=True, default="Ag/AgCl")

    def __post_init__(self):
        self.data = self.data.rename(columns={"freq/Hz":"freq", "Re(Z)/Ohm": "real", "-Im(Z)/Ohm": "imag", "|Z|/Ohm": "comp", "Phase(Z)/deg": "phase","time/s":"t", 
    'cyle number':'cycle', "time/s":"t", "Ewe/V": "E", "<I>/mA": "I", "cycle number": "cycle", "<Ewe>/V": "Ewe","control/mA":"Icontrl", "control/V":"Econtrl"})
        self.columns = self.data.columns
        self.cycles = self.data.cycle.unique()
        

    def change_units(self, timeunits='s', Iunits='mA'):
        if (self.timeunits == timeunits) & (self.Iunits == Iunits):
            return
        if timeunits == 'm':
            if self.timeunits == 's':
                self.data.t = self.data.t / 60
                self.timeunits = 'm'
        else:
            if self.timeunits == 'm':
                self.data.t = self.data.t * 60
                self.timeunits = 's'     
        if Iunits == 'A':
            self.data.I = self.data.I / 1000
            self.data.Icontrl = self.data.Icontrl / 1000

    def plot(self, timeunit='s', save=False, figpass = "", cycles=False, fname='fig.png'):
        if figpass == "":
            fig = plt.figure()
        else:
            fig = plt.figure(figpass)
        if cycles == False:
            plt.plot(self.data.t, self.data.I)
            plt.xlabel("Time (%s)" % self.timeunits)
            plt.ylabel("I (%s)" % (self.Iunits))
            
        else:
            for cycle in self.cycles:
                d = self.data[self.data.cycle == cycle]
                plt.plot(d.t, d.I, marker='.', linestyle="None")
                
            plt.xlabel("Time (%s)" % self.timeunits)
            plt.ylabel("I (%s)" % (self.Iunits))
        if save == True:
            plt.savefig(fname, format='png', dpi=300)
        plt.show()

        return fig