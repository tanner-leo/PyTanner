#import pandas as pd
#pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
#import os
import cmath
from dataclasses import dataclass, field
import pandas as pd
from scipy import optimize


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
    





def plt_bode(df, fname='./Figure_bode.png', legends=[""], save=False, title='', markersize=6, bbox=[0.95,0.8]):
    
    if legends == [""]:
        fig, ax = plt.subplots(1,1)
        ax.plot(df.freq, abs(df.comp), 'black', marker='x', label='|Z|', markersize=markersize)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax2 = ax.twinx()
        ax2.plot(df.freq, df.phase, 'red', marker='o', label='$f$', markersize=markersize)
        ax2.set_ylabel("Phase Angle (deg)")
        ax.set_ylabel("|$Z$| (Ohms)")
        ax.set_xlabel("Frequency (Hz)")
        ax2.spines['right'].set_color('red')
        ax2.spines['left'].set_color('black')
        fig.legend(bbox_to_anchor=(1.05, 1))
        
        fig.tight_layout()
        if save == True:
            fig.savefig(fname, format='png', dpi=300, bbox_inches='tight')
        return fig
    else:
        linestyles = ['x', '.', 'o', '-*']
        fig, ax = plt.subplots(1,1)
        for self, label, linestyle in zip(df, legends, linestyles):
            ax.plot(self.freq, abs(self.comp), 'black', marker=linestyle, label="|Z| %s" % label, linewidth=1, markersize=markersize)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax2 = ax.twinx()
        for self, label, linestyle in zip(df, legends, linestyles):
            ax2.plot(self.freq, self.phase, 'red', marker=linestyle, linewidth=1, markersize=markersize, label="$f$ %s" % label)
        #ax2.set_ylim(-90, 90)
        ax2.set_ylabel("Phase Angle (deg)")
        ax.set_ylabel("|$Z$| (Ohms)")
        ax.set_xlabel("Frequency (Hz)")
        ax2.spines['right'].set_color('red')
        ax2.spines['left'].set_color('black')
        fig.legend(bbox_to_anchor=(bbox[0], bbox[1]),loc='lower left')

        fig.tight_layout()
        if save == True:
            fig.savefig(fname, format='png', dpi=300, bbox_inches='tight')
        return fig

def plt_nyquist(df, legends=[""], save=False, fname="./Figure_nyquist.png", title=""):
    if legends != [""]:
        for df, label in zip(df, legends):
            plt.plot(df.real, df.imag, 'o', markersize=3, label=label)
            plt.axis('square')
            plt.xlim(-1,max(df.real)*1.1)
            plt.ylim(-1,max(df.real)*1.1)
        plt.legend()
    else:
        for df in df:
            plt.plot(df.real, df.imag, 'o', markersize=3)
            plt.axis('square')
            plt.xlim(-1,max(df.real)+5)
            plt.ylim(-1,max(df.real)+5)
    plt.ylabel('-Z$_{imag}$ (ohms)')
    plt.xlabel('Z$_{real}$ (ohms)')
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
        # def plt_nyquist(df, save=False, fname="./Figure_nyquist.png", title=""):
        plt.plot(self.real, self.imag, 'ro', markersize=3)
        plt.axis('square')
        plt.xlim(-1,max(self.real)*1.1)
        plt.ylim(-1,max(self.real)*1.1)
        plt.xlabel("$Z_{real}$ (ohms)")
        plt.ylabel("$Z_{imag}$ (ohms)")
        plt.tight_layout()
        if title != "":
            plt.title(title)
        plt.tight_layout()
        if save == True:
            plt.savefig(fname, format='png', dpi=300, bbox_inches='tight')
        plt.show()

    def bode(self, save=False, fname='./Fig.png', title=''):
        plt_bode(self.data, save=save, fname=fname, title=title)
    
    def dual_plots(self, ylim=[0,1000], fw=10, fname='./Figure.png', save=False):
        plt_dual_plots(self.data, ylim=ylim, fw=fw, fname=fname, save=save)


    def write_description(self, descrp):
        self.description = descrp

    def fitting(self):
        fitter = PEISfit(self.data)
        fitter.fit()
        fitter.plot()


    

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
        if timeunits == 'min':
            if self.timeunits == 's':
                self.data.t = self.data.t / 60
                self.timeunits = 'min'
        else:
            if self.timeunits == 'min':
                self.data.t = self.data.t * 60
                self.timeunits = 's'     
        if Eunits == 'mV':
            self.data.E = self.data.E *1000
            self.data.Econtrl = self.data.Econtrl * 1000


    def plot(self, variable='E', timeunit='s', save=False, figpass =False, cycles=False, fname='./Fig.png', line=True, markersize=4):
        
        if cycles == False:
            if line==False:
                plt.plot(self.data.t, self.data.E, linestyle="None", marker='o', markersize=markersize)
            else:
                plt.plot(self.data.t, self.data.E)
            plt.xlabel("Time (%s)" % self.timeunits)
            plt.ylabel("E (%s vs %s)" % (self.Eunits, self.reference))
            
        else:
            for cycle in self.cycles:
                d = self.data[self.data.cycle == cycle]
                plt.plot(d.t, d.E)
                # plt.plot(d.t, d.E, marker='.', linestyle="None") #alternative plotting method
                
            plt.xlabel("Time (%s)" % self.timeunits)
            plt.ylabel("I (%s)" % self.Eunits)
        if save == True:
            plt.savefig(fname, format='png', dpi=300)
        if figpass == True:
            plt.show(block=True)
            return

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
        if timeunits == 'min':
            if self.timeunits == 's':
                self.data.t = self.data.t / 60
                self.timeunits = 'min'
        else:
            if self.timeunits == 'min':
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

        #return fig

@dataclass
class OCV:
    data: pd.DataFrame
    name: str = field(repr=True, default="CA")
    description: str = field(repr=False, default="description")
    timeunits: str = 's'
    Eunits: str = "V"
    reference: str = field(repr=True, default="Ag/AgCl")

    def __post_init__(self):
        if self.timeunits != 's':
            self.add_units(timeunits=self.timeunits)

    
    def make_cycles(self, threshold=50):
        self.data['dif'] = self.data.t.diff()
        self.data['cycle'] = np.nan
        cycle = 1
        for index, row in self.data.iterrows():
            if row.dif > 50:
                cycle=cycle+1
            row.cycle = cycle
            
            self.data.iloc[index] = row

    def plot_cycles(self, timeunit=timeunits , save=False, figpass = "", fname='./Fig.png', line=True, markersize=4):
        max, tmax = [], []
        for cycle in self.data.cycle.unique():
            m = self.data[self.data.cycle == cycle]
            if timeunit == 's':
                plt.plot(m.t, m.E, marker='.', linestyle="None", label="Cycle %i" % cycle)
            elif timeunit == 'min':
                plt.plot(m['t(min)'], m.E, marker='.', linestyle="None", label="Cycle %i" % cycle)
            max.append(m[m.t == m.t.max()].E.values[0])
            tmax.append(m.t.max())
        plt.legend()
        plt.xlabel("Time (%s)" % self.timeunits)
        plt.ylabel("E (%s vs %s)" % (self.Eunits, self.reference))
        self.cycle_maxE = max
        self.cycle_maxt = tmax


    def add_units(self, timeunits='', Eunits=''):
        if timeunits == 'min':
            self.data['t(min)'] = self.data.t / 60
        if Eunits == 'mV':
            self.data['E(mV)'] = self.data.E*1000





def Zcap(omega,Cd):
    #j = np.sqrt(-1)
    return -1j/(2*np.pi*omega*Cd)

def Zres(R1):
    return R1

def Z_rc_s(omega, Cd, R1): # series circuit of R and C
    return Zres(R1) + Zcap(omega, Cd)
def Z_rc_s_abs(omega, Cd, R1): # series circuit of R and C
    return abs(Zres(R1) + Zcap(omega, Cd))

def Z_rq_p(omega, R2, Q2, a2):
    return Zres(R2)/(Zres(R2)*Q2*(1j*2*np.pi*omega)**a2+1)

def Z_rc_p(omega, Cd, R1): # paralell circuit of R and C
    return (Zres(R1)**-1+Zcap(omega, Cd)**-1)**-1

def Z_randles(omega, Cd, R1, R2): # Randles Circuit
    return Zres(R2) + Z_rc_p(omega, Cd, R1)

def Z_randles_p(omega, Cd1, R1, R2, Cd2, R3, R4): # R2 and R4 are series with the Cd1/R1, and Cd2/R3
    return ((Zres(R2) + Z_rc_p(omega, Cd1, R1))**-1 + (Zres(R4) + Z_rc_p(omega, Cd2, R3))**-1)**-1

def Z_randles_s(omega, Cd1, R1, R2, Cd2, R3): # R2 and R4 are series with the Cd1/R1, and Cd2/R3
    #omega = omega/(2*np.pi)
    return Zres(R2) + Z_rc_p(omega, Cd1, R1) + Z_rc_p(omega, Cd2, R3)

def Z_randles_s(omega, Cd1, R1, R2, Cd2, R3): # R2 and R4 are series with the Cd1/R1, and Cd2/R3
    #omega = omega/(2*np.pi)
    return Zres(R2) + Z_rc_p(omega, Cd1, R1) + Z_rc_p(omega, Cd2, R3)

def Z_randles_s_pseudo(omega, R1, Q2, R2, Q3, R3, a2, a3):
    return Zres(R1) + Z_rq_p(omega, R2, Q2, a2) + Z_rq_p(omega, R3, Q3, a3)

@dataclass
class PEISfit:
    dc: dataclass
    p0: tuple = field(repr=False, default="None")
    fitfunction: str = field(repr=False, default='Z_randles_s_pseudo')
    

    def __post_init__(self):
        self.data = self.dc.data
        self.functions = ['Z_rc_p', 'Z_randles_p', 'Z_randles_s', 'Z_randles_s_pseudo']
        if self.fitfunction == 'Z_randles_s_pseudo':
            if self.p0 == "None":
                self.p0 =(10, 100., 1e-6, 100., 1e-6, 0.5, 0.5)
            columns=("R1", "Q2", "R2", "Q3", "R3", "a2", "a3")
            temp = [self.p0]
            self.params = pd.DataFrame(temp, columns=columns)
            self.params.rename(index={0:'initial guess'}, inplace=True)
            

    def fit(self, cycle=1):
        df1 = self.data[(self.data.cycle == cycle)&(self.data.imag > 0)]
        
        if self.fitfunction == 'Z_randles_s':
            p0=(10e-6, 1000, 16, 10e-6, 10000)
            def f(omega, Cd1, R1, R2, Cd2, R3): 
                return np.hstack([np.real(Z_randles_s(omega, Cd1, R1, R2, Cd2, R3)), np.imag(np.real(Z_randles_s(omega, Cd1, R1, R2, Cd2, R3)))]) #fitting function
            xdata, ydata = df1.freq, np.hstack([df1.real, df1.imag]) # assigning fitting variables
            popt, pcov = optimize.curve_fit(f, xdata, ydata, p0=p0, bounds=(0,10000),
                               method='trf') #perform fit
            eva = Z_randles_s(xdata, *popt) #evaluating function without abs
            self.residual = ydata - eva
            self.xaxis=xdata
            real, imag = [d.real for d in eva], [d.imag * -1 for d in eva] #evaluating
            self.fitresult = pd.DataFrame({'real':real, 'imag':imag, 'comp':abs(eva), 'freq':xdata, 'phase':np.angle(eva, deg=True)}) #assigning df
            self.fitdata = df1

        elif self.fitfunction == 'Z_randles_s_pseudo':
            p0 = self.p0
            #p0=(10, 100., 1e-6, 100., 1e-6, 0.5, 0.5)
            def f1(omega, R1, Q2, R2, Q3, R3, a2, a3): 
                solve = Z_randles_s_pseudo(omega, R1, Q2, R2, Q3, R3, a2, a3)
                #real, imag = [d.real for d in solve], [d.imag * -1 for d in solve]
                real = np.real(solve)
                imag = np.imag(solve)*-1
                return np.hstack([real, imag])
            xdata, ydata = df1.freq, np.hstack([df1.real, df1.imag]) # assigning fitting variables
            popt, pcov = optimize.curve_fit(f1, xdata, ydata, p0=p0, bounds=([0, -10, 0, -10, 0, 0, 0],[1000, 1000, 1000, 1000, 1000, 1, 1]), max_nfev=10000,
                            method='trf') #perform fit
            xf = f1(xdata, *popt)
            eva = Z_randles_s_pseudo(xdata, *popt) #evaluating function
            real, imag = [d.real for d in eva], [d.imag * -1 for d in eva] #evaluating
            #creating residuals
            self.residual1 = df1.real - real
            self.residual2 = df1.imag - imag
            self.xaxis=xdata
            self.SSE = np.sum(self.residual1**2)+np.sum(self.residual2**2) # Sum of squares error
            self.SS = np.sum((real - np.average(real))**2)
            dfmodel = len(p0)
            dfdata = len(real)*2
            self.R2_adj = 1-((self.SSE/self.SS)*(dfdata/dfmodel))
                        
            self.fitresult = pd.DataFrame({'real':real, 'imag':imag, 'comp':abs(eva), 'freq':xdata, 'phase':np.angle(eva, deg=True)}) #assigning df
            self.fitdata = df1
            param_labels = np.array(["R1", "Q2", "R2", "Q3", "R3", "a2", "a3"])
            
            temp = [popt]
            temp2 = pd.DataFrame(temp, columns=param_labels, index=[cycle])
            temp2["R^2"] = self.R2_adj
            self.params = pd.concat([self.params, temp2])
            self.pcov = pcov
            self.perr = np.sqrt(np.diag(pcov))

        elif self.fitfunction == 'Z_randles_s_pseudo-free':
            p0 = self.p0
            #p0=(10, 100., 1e-6, 100., 1e-6, 0.5, 0.5)
            def f1(omega, R1, Q2, R2, Q3, R3, a2, a3): 
                solve = Z_randles_s_pseudo(omega, R1, Q2, R2, Q3, R3, a2, a3)
                #real, imag = [d.real for d in solve], [d.imag * -1 for d in solve]
                real = np.real(solve)
                imag = np.imag(solve)*-1
                return np.hstack([real, imag])
            xdata, ydata = df1.freq, np.hstack([df1.real, df1.imag]) # assigning fitting variables
            popt, pcov = optimize.curve_fit(f1, xdata, ydata, method='lm') #perform fit
            # xf = f1(xdata, *popt)
            eva = Z_randles_s_pseudo(xdata, *popt) #evaluating function
            real, imag = [d.real for d in eva], [d.imag * -1 for d in eva] #evaluating
            # Residuals
            self.residual1 = df1.real - real
            self.residual2 = df1.imag - imag
            self.xaxis=xdata
            self.SSE = np.sum(self.residual1**2)+np.sum(self.residual2**2) # Sum of squares error
            self.SS = np.sum((real - np.average(real))**2)
            dfmodel = len(p0)
            dfdata = len(real)*2
            self.R2_adj = 1-((self.SSE/self.SS)*(dfdata/dfmodel))

            self.fitresult = pd.DataFrame({'real':real, 'imag':imag, 'comp':abs(eva), 'freq':xdata, 'phase':np.angle(eva, deg=True)}) #assigning df
            self.fitdata = df1
            param_labels = np.array(["R1", "Q2", "R2", "Q3", "R3", "a2", "a3"])
            temp = [popt]
            temp2 = pd.DataFrame(temp, columns=param_labels, index=[cycle])
            temp2["R^2"] = self.R2_adj
            self.params = pd.concat([self.params, temp2])
            self.pcov = pcov
            self.perr = np.sqrt(np.diag(pcov))

        
    def fit_cycles(self, plot=False, residuals=False):
        cycles = self.data.cycle.unique()
        for cycle in cycles:
            self.fit(cycle=cycle)
            if plot == True:
                self.plot()
            if residuals==True:
                self.plot_residuals()
    
    def plot_residuals(self):
        plt.plot(self.xaxis, self.residual1, label="Real Residuals")
        plt.plot(self.xaxis, self.residual2, label="Imaginary Residuals")
        plt.xscale('log')
        plt.legend()
        plt.ylabel("Residuals (ohms)")
        plt.xlabel('log frequency (Hz)')
        plt.show()
    


    def plot(self, type='nyquist', save=False, fname="Fig.png"):
        try:
            dd = self.fitresult
            df1 = [self.data.real, self.data.imag]
        except:
            print("No Fit Performed!")
            return
        else:
            dd = self.fitresult
            df1 = self.data
            if type == 'bode':
                plt_bode([dd,df1], legends=['fit', 'data'], fname=fname, save=False) # plot results
            if type == 'nyquist':
                plt_nyquist([dd, df1], legends=['fit', 'data'], fname=fname, save=False)
