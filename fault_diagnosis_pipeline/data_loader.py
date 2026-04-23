import numpy as np

def generate_vibration_data(duration=1.0, fs=10000, condition="healthy", rpm=1800, noise_level=0.5):
    """
    Generate mock raw vibration data for a rotating machine.
    
    Args:
        duration (float): Time duration in seconds.
        fs (int): Sampling frequency in Hz.
        condition (str): "healthy", "imbalance" (1X), "misalignment" (2X), or "bearing" (inner race fault).
        rpm (float): Rotation per minute.
        noise_level (float): Amplitude of Gaussian noise.
        
    Returns:
        t (np.ndarray): Time vector.
        signal (np.ndarray): Vibration signal vector.
    """
    t = np.arange(0, duration, 1/fs)
    signal = np.random.normal(0, noise_level, len(t))
    
    # Fundamental frequency (1X)
    f_1x = rpm / 60.0
    
    # Healthy machine always has some fundamental frequency component
    signal += 1.0 * np.sin(2 * np.pi * f_1x * t)
    
    if condition == "imbalance":
        # Imbalance adds a large spike at 1X
        signal += 5.0 * np.sin(2 * np.pi * f_1x * t)
    elif condition == "misalignment":
        # Misalignment usually shows strong 2X and sometimes 3X
        f_2x = 2 * f_1x
        signal += 3.5 * np.sin(2 * np.pi * f_2x * t)
        signal += 1.5 * np.sin(2 * np.pi * 3 * f_1x * t)
    elif condition == "bearing":
        # Bearing fault (e.g., BPFI) might show up at a specific non-integer multiple of 1X.
        # Let's say BPFI is 5.43 times the fundamental frequency.
        bpfi = 5.43 * f_1x
        signal += 2.8 * np.sin(2 * np.pi * bpfi * t)
        
        # Adding some harmonics of bearing fault
        signal += 1.0 * np.sin(2 * np.pi * 2 * bpfi * t)
        signal += 0.5 * np.sin(2 * np.pi * 3 * bpfi * t)
        
    return t, signal

if __name__ == "__main__":
    t, sig = generate_vibration_data(condition="imbalance")
    print(f"Generated signal of length {len(sig)} for 1 second at 10kHz.")
