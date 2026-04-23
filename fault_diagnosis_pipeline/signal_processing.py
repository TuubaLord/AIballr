import numpy as np
import scipy.fft as fft
from scipy.signal import find_peaks

def compute_fft(t, signal):
    """
    Compute the FFT of a real time-domain signal.
    
    Args:
        t (np.ndarray): Time vector.
        signal (np.ndarray): Amplitude vector.
        
    Returns:
        freqs (np.ndarray): Frequency vector (positive half only).
        magnitude (np.ndarray): Magnitude of FFT components.
    """
    N = len(t)
    dt = t[1] - t[0]
    
    # Compute FFT
    yf = fft.rfft(signal)
    xf = fft.rfftfreq(N, d=dt)
    
    # Compute magnitude
    magnitude = np.abs(yf) / (N / 2) # normalize
    # DC component shouldn't be multiplied by 2 like the others
    magnitude[0] = magnitude[0] / 2
    
    return xf, magnitude

def extract_peaks(freqs, magnitude, height_threshold=None, exclude_dc=True):
    """
    Find peaks in the amplitude spectrum.
    
    Args:
        freqs (np.ndarray): Frequency vector.
        magnitude (np.ndarray): Amplitude spectrum.
        height_threshold (float): Minimum height for a peak to be considered.
        exclude_dc (bool): Exclude DC component (0 Hz).
        
    Returns:
        peak_freqs (list): Frequencies where peaks were found.
        peak_mags (list): Magnitudes of those peaks.
    """
    if height_threshold is None:
        # Dynamic threshold based on relative magnitude sizes
        height_threshold = max(0.05, np.max(magnitude) * 0.1)
        
    peaks, properties = find_peaks(magnitude, height=height_threshold)
    
    peak_freqs = freqs[peaks]
    peak_mags = properties["peak_heights"]
    
    filtered_peaks = []
    for f, m in zip(peak_freqs, peak_mags):
        if exclude_dc and f < 5.0:  # Skip frequencies extremely close to zero to handle windowing artifacts
            continue
        filtered_peaks.append((f, m))
        
    return [p[0] for p in filtered_peaks], [p[1] for p in filtered_peaks]

def compare_to_baseline(baseline_t, baseline_sig, test_t, test_sig):
    """
    Compare test signal to baseline and extract significant peaks.
    
    Returns:
        basline_peaks (list of tuples): (freq, mag)
        test_peaks (list of tuples): (freq, mag)
        differences (list): list of strings describing new or increased peaks
    """
    f_b, m_b = compute_fft(baseline_t, baseline_sig)
    f_t, m_t = compute_fft(test_t, test_sig)
    
    bp_f, bp_m = extract_peaks(f_b, m_b, height_threshold=None)
    tp_f, tp_m = extract_peaks(f_t, m_t, height_threshold=None)
    
    baseline_peaks = list(zip(bp_f, bp_m))
    test_peaks = list(zip(tp_f, tp_m))
    
    # Simple logic for finding new or amplified peaks
    differences = []
    for tf, tm in test_peaks:
        # See if there's a corresponding peak in baseline near this freq
        found = False
        for bf, bm in baseline_peaks:
            if abs(tf - bf) < 2.0: # 2 Hz tolerance
                found = True
                if tm > bm * 1.5:
                    differences.append({"freq": float(np.round(tf, 2)), "mag": float(np.round(tm, 2)), "status": f"amplified (from {np.round(bm, 2)})"})
                break
        
        if not found:
            differences.append({"freq": float(np.round(tf, 2)), "mag": float(np.round(tm, 2)), "status": "new peak"})
            
    return baseline_peaks, test_peaks, differences

if __name__ == "__main__":
    from data_loader import generate_vibration_data
    
    # Quick test
    tb, sb = generate_vibration_data(condition="healthy")
    tt, st = generate_vibration_data(condition="misalignment")
    
    bp, tp, diffs = compare_to_baseline(tb, sb, tt, st)
    print("Detected anomalies:", diffs)
