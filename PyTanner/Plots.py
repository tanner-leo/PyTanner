#import pandas as pd
#pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
#import os
import cmath


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
