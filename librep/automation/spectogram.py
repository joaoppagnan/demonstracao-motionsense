import numpy as np
from scipy import signal
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

class ExtractSpectogram:
    def __init__(self, fs: float, nperseg: int, nooverlap: int):
        self.fs = fs
        self.nperseg = nperseg
        self.nooverlap = nooverlap
        self.f = None
        self.t = None
        self.s = None
        
    def run(self, input_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        f, t, s = signal.spectrogram(input_data, fs=self.fs, nperseg=self.nperseg, noverlap=self.nooverlap)
        self.f = f
        self.t = t
        self.s = s
        return (f, t, s)
    
class GetSpectogramStatistics:
    def __init__(self):
        self.s_min = None
        self.s_max = None
        self.s_avg = None
        self.s_std = None
        
    
    def run(self, input_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[float, float, float, float]:
        f, t, s = input_data
        self.s_min = np.min(s)
        self.s_max = np.max(s)
        self.s_avg = np.mean(s)
        self.s_std = np.std(s)
        return self.s_min, self.s_max, self.s_avg, self.s_std
    
class PlotMultiSpectogram:
    def __init__(self, figtitle: str, 
                 nrows: int, 
                 ncols: int, 
                 row_names: List[str], 
                 col_names: List[str], 
                 xlabel: str, 
                 ylabel: str, 
                 vmin: float = 0.0, 
                 vmax: float = 3.0, 
                 figsize=None, 
                 shading='gouraud',
                 sharex=False,
                 sharey=False):
        self.figtitle = figtitle
        self.nrows = nrows
        self.ncols = ncols
        self.row_names = row_names
        self.col_names = col_names
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.vmin = vmin
        self.vmax = vmax
        self.figsize = figsize
        self.shading = shading
        self.sharex = sharex
        self.sharey = sharey
        self.fig = None
        self.axs = None
    
    def run(self, input_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], output_path: str = None):
        figsize = (self.ncols*3, self.nrows*2)
        fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=figsize, squeeze=False, sharex=self.sharex, sharey=self.sharey)
        #plt.rcParams['figure.constrained_layout.use'] = True        
        self.fig = fig
        self.axs = axs
        plt.subplots_adjust(top=0.80, hspace=0.9, wspace=0.2)
        fig.suptitle(self.figtitle, y=0.85, x=0.5, fontsize=18)
        for i in range(self.nrows):
            for j in range(self.ncols):
                data = input_data[i][j]
                if data is None:
                    continue
                axs_title = f"{self.row_names[i]} ({self.col_names[j]})"                
                axs[i,j].set_ylabel(self.ylabel)
                axs[i,j].set_xlabel(self.xlabel)
                axs[i,j].set_title(axs_title)
                axs[i,j].pcolormesh(data[1], data[0], data[2], shading=self.shading, vmin=self.vmin, vmax=self.vmax)
                
        
        #fig.tight_layout(pad=self.pad)
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', transparent=False)
        plt.show()
        plt.close()