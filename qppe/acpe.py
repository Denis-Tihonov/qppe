import numpy as np
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf

def AutoCorrPhaseIdentifier(x, period):
    
    acf_result = acf(x[:], nlags = len(x), fft=True)
     
    acf_indeces_phase = find_peaks(acf_result, distance = int(period * 0.75))[0]
    
    acf_indeces_phase = np.append(np.array([0]), acf_indeces_phase)
    phase = np.array([])
    for i in range(len(acf_indeces_phase)-1):
        i_start = acf_indeces_phase[i]
        i_end = acf_indeces_phase[i+1]
    
        phase = np.append(phase, np.linspace(0, 2*np.pi, int(i_end - i_start)))
    
    period_end = len(x) - acf_indeces_phase[-1]
    
    phase = np.append(phase, np.linspace(0, 2 * np.pi * period_end/int(i_end - i_start), period_end))
    return phase