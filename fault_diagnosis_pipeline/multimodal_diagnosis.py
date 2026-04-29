import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, detrend, stft, butter, sosfiltfilt, find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import requests
import base64
import os
import re

# 1. Data Ingestion & Kinematics (Programmatic)
def load_mat_file(filepath):
    """Loads a CWRU .mat file. Returns the DE (Drive End) time series signal."""
    data = sio.loadmat(filepath)
    # Find the DE time key dynamically
    de_key = next((k for k in data.keys() if 'DE' in k), None)
    if de_key is None:
        raise ValueError("Could not find 'DE' key in mat file.")
    signal = data[de_key].flatten()
    return signal

import numpy as np

def calculate_kinematics(rpm, location="DE", contact_angle=0):
    """
    Calculate theoretical fault frequencies based on physical bearing geometry.
    Supports CWRU Drive End (SKF 6205) and Fan End (SKF 6203) bearings.
    
    rpm: Shaft speed in RPM
    location: "DE" for Drive End, "FE" for Fan End
    contact_angle: Default 0 degrees for standard deep groove ball bearings
    """
    f_r = rpm / 60.0
    
    # 1. Dynamically assign bearing geometry based on location
    if location == "DE":
        # SKF 6205-2RS JEM (Drive End)
        pitch_diameter = 1.537
        ball_diameter = 0.3126
        nb = 9
    elif location == "FE":
        # SKF 6203-2RS JEM (Fan End)
        pitch_diameter = 1.122
        ball_diameter = 0.2656
        nb = 8
    else:
        raise ValueError("Location must be 'DE' or 'FE'")

    # 2. Transparent Kinematic Math
    bd_pd = ball_diameter / pitch_diameter
    cos_alpha = np.cos(np.radians(contact_angle))
    
    bpfo = (nb / 2.0) * f_r * (1 - bd_pd * cos_alpha)
    bpfi = (nb / 2.0) * f_r * (1 + bd_pd * cos_alpha)
    bsf = (pitch_diameter / (2.0 * ball_diameter)) * f_r * (1 - (bd_pd * cos_alpha)**2)
    #ftf = 0.5 * f_r * (1 - bd_pd * cos_alpha)
    
    return {
        "1X": f_r,
        "BPFO": bpfo,
        "BPFI": bpfi,
        "BSF": bsf,
        #"FTF": ftf
    }

def plot_time_series(signal, fs=12000, image_path="raw_time_series.png", title="Raw Time Series Signal"):
    """
    Plots the raw time domain series of the signal.
    """
    t = np.arange(len(signal)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal, color='darkblue', linewidth=1.0)
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

def plot_master_envelope(xf, magnitude, kinematics, image_path, title):
    """
    Creates a high-fidelity 0-500Hz master diagnostic overview plot with all fault markers.
    """
    plt.figure(figsize=(15, 6))
    plt.plot(xf, magnitude, color="teal", linewidth=1.2, label="Envelope Magnitude", zorder=5)
    
    # Fault Marker Configuration
    markers = [
        ("1X", "blue", kinematics["1X"]),
        ("BPFO", "red", kinematics["BPFO"]),
        ("BPFI", "green", kinematics["BPFI"]),
        ("BSF", "orange", kinematics["BSF"]),
    ]
    
    for label, color, freq in markers:
        plt.axvline(x=freq, color=color, linestyle="--", alpha=0.7, linewidth=2.0, label=f"{label} ({freq:.1f} Hz)", zorder=1)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Magnitude", fontsize=12)
    plt.xlim(0, 500)
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

# 2. Phase 1, 2, 3: Envelope Analysis & Image Generation (Programmatic)
def run_envelope_analysis(signal, fs, fault_freqs, target_fault=None, image_path="envelope_spectrum.png", phase_title="Phase 1: Squared Envelope Spectrum"):
    """
    Computes the Squared Envelope Spectrum.
    """
    sig_detrend = detrend(signal)
    analytic_signal = hilbert(sig_detrend)
    amplitude_envelope = np.abs(analytic_signal)
    squared_envelope = amplitude_envelope**2
    N = len(squared_envelope)
    yf = rfft(squared_envelope - np.mean(squared_envelope))
    xf = rfftfreq(N, 1/fs)
    magnitude = np.abs(yf) / (N/2)  # normalize
    
    # Extract structural peaks context
    peaks, _ = find_peaks(magnitude, height=np.mean(magnitude)*3, distance=10)
    if len(peaks) > 0:
        top_peak_freqs = xf[peaks[np.argsort(magnitude[peaks])][-5:]]
        top_peak_freqs_str = ", ".join([f"{f:.1f}" for f in np.sort(top_peak_freqs)])
    else:
        top_peak_freqs_str = "None"
    numerical_context = f"The top 5 highest energy peaks in this spectrum occur at exactly: [{top_peak_freqs_str}] Hz. Compare these to the theoretical fault frequencies."
    
    encoded_images = []
    
    # Scale subimages to absolute global envelope max to prevent auto-scaling micro-peaks
    global_max = np.max(magnitude) if len(magnitude) > 0 else 1.0
    ylim_upper = global_max * 1.1
    
    # Plotting harmonic zooms if target_fault is provided
    if target_fault and target_fault in fault_freqs:
        base_freq = fault_freqs[target_fault]
        true_freq = base_freq
        
        for i, h in enumerate([1, 2, 3]):
            if h == 1:
                # Drift Correction: Snap fundamental frequency onto local maximum peak inside +-2Hz window
                search_mask = (xf >= max(0, base_freq - 2.0)) & (xf <= base_freq + 2.0)
                if np.any(search_mask):
                    peak_idx = np.argmax(magnitude[search_mask])
                    true_freq = xf[search_mask][peak_idx]
            
            target_x = true_freq * h
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Zoom logic: +/- 15Hz around target
            zoom_margin = 8
            idx_range = (xf >= max(0, target_x - zoom_margin)) & (xf <= target_x + zoom_margin)
            
            if np.any(idx_range):
                ax.plot(xf[idx_range], magnitude[idx_range], color="teal", linewidth=2.5, zorder=2)
            ax.axvline(x=target_x, color="red", linestyle="--", alpha=1.0, linewidth=3.5, zorder=1)
            ax.set_ylim(0, ylim_upper)
            ax.set_xticks([])
            ax.set_yticks([])
                
            plt.tight_layout()
            sub_img_path = image_path.replace(".png", f"_{h}x.png")
            plt.savefig(sub_img_path, bbox_inches='tight')
            plt.close()
            
            with open(sub_img_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append((encoded_string, sub_img_path, h))
                
        return encoded_images, numerical_context
    else:
        # standard plot if no target fault
        plt.figure(figsize=(10, 4))
        plt.plot(xf, magnitude, label="Envelope Spectrum", color="teal", linewidth=2.5)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()
        
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return [(encoded_string, image_path, 1)], numerical_context

# 3. Phase 2 & 3: Prewhitening & Benchmark
def run_cepstrum_prewhitening(signal):
    """
    Method 2: Cepstrum Prewhitening (Phase-only reconstruction).
    Flattens the magnitude spectrum to 1, retaining only phase information,
    which strongly highlights impulsive bearing faults over continuous deterministic noise.
    """
    yf = rfft(signal)
    phase = np.angle(yf)
    whitened_yf = np.exp(1j * phase) # Magnitude set to 1
    whitened_signal = np.fft.irfft(whitened_yf, n=len(signal))
    return whitened_signal

def run_spectral_kurtosis_and_filter(signal, fs, image_path="spectral_kurtosis.png"):
    """
    Method 3: Spectral Kurtosis.
    Computes SK via STFT to find the frequency band with the highest impulsiveness,
    filters the signal within that optimal band, and returns it.
    """
    f, t, Zxx = stft(signal, fs=fs, nperseg=512)
    mag = np.abs(Zxx)
    sk = kurtosis(mag, axis=1, fisher=False)
    
    # Save a diagnostic SK plot
    plt.figure(figsize=(10, 4))
    plt.plot(f, sk, color='purple')
    plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    
    # Filter original signal around peak SK
    peak_idx = np.argmax(sk)
    optimal_fc = f[peak_idx]
    
    # Create bandpass filter (1000 Hz bandwidth around fc)
    bw = 1000.0
    lowcut = max(5.0, optimal_fc - bw/2)
    highcut = min(fs/2 - 5.0, optimal_fc + bw/2)
    
    nyq = 0.5 * fs
    # Use Second-Order Sections (SOS) mapping for numerical stability near Nyquist
    sos = butter(4, [lowcut/nyq, highcut/nyq], btype='band', output='sos')
    filtered_signal = sosfiltfilt(sos, signal)
    
    return filtered_signal, optimal_fc, image_path

# 4. The "Human" Visual Evaluation (Local Multimodal LLM via Ollama)
def parse_llm_diagnosis(response_text):
    if ":::" in response_text:
        return response_text.split(":::")[0].upper().strip()
        
    text = response_text.upper()
    if "HEALTHY" in text:
        return 'HEALTHY'
    elif "UNDIAGNOSABLE" in text:
        return 'UNDIAGNOSABLE'
    
    fault_type = 'UNKNOWN_FAULT'
    if 'BALL' in text or 'BSF' in text or 'ROLLING ELEMENT' in text:
        fault_type = 'BALL'
    elif 'INNER' in text or 'BPFI' in text or 'IR' in text:
        fault_type = 'INNER_RACE'
    elif 'OUTER' in text or 'BPFO' in text or 'OR' in text:
        fault_type = 'OUTER_RACE'
    return fault_type

def evaluate_single_harmonic_with_llm(b64_image, image_filename, numerical_context, target_fault, harmonic, phase_name):
    """
    Sends the specific generated harmonic plot to Gemma for diagnostic evaluation.
    """
    prompt_string = (
        f"- If a massive peak perfectly aligns directly on top of the red dashed line: 1\n"
        "- If the peak is misaligned, ambiguous, or just background noise: 0\n"
        "- If there is NO peak, or the signal is totally flat at that marker: -1\n"
        "CRITICAL RULE: Assume NO PEAK by default. Only award 1 if the alignment is exceptionally obvious and undeniable. You must penalize (-1) for missing peaks. Do NOT hallucinate data.\n"
        "Output your final evaluation EXACTLY in this format on the very last line:\n"
        "SCORE: [Score from -1 to 1]"
    )
    
    print(f"========== SUPERVISION: DATA SENT TO AI ({phase_name} / {target_fault} / {harmonic}x) ==========")
    print(f"Image Attached: {image_filename}")
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma4",
        "prompt": prompt_string,
        "images": [b64_image],
        "stream": False,
        "options": {
            'temperature': 0
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response content found.")
    except Exception as e:
        return f"Error connecting to Ollama check if server is running: {e}"

def run_full_diagnosis_pipeline(target_mat_file, location=None, phase_1_only=False):
    """
    Executes the complete multi-modal sequential specific fault isolation pipeline.
    Returns the mapped string fault output concatenated with the diagnosis reasoning.
    """
    if not os.path.exists(target_mat_file):
        raise FileNotFoundError(f"Dataset file {target_mat_file} not found.")
        
    print(f"Loading data from {target_mat_file}...")
    sig_test = load_mat_file(target_mat_file)
    
    fs = 12000
    
    # Calculate Dynamic RPM based on Motor Load (HP) digit in filename
    basename = os.path.basename(target_mat_file).replace('.mat', '')
    try:
        hp = int(basename.split('_')[-1])
        hp_map = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
        theoretical_rpm = hp_map.get(hp, 1797)
    except ValueError:
        theoretical_rpm = 1797
        
    # # Find True RPM via Raw FFT peak hunting near theoretical 1X (running speed)
    # theoretical_1x = theoretical_rpm / 60.0
    # N = len(sig_test)
    # raw_yf = rfft(sig_test - np.mean(sig_test))
    # raw_xf = rfftfreq(N, 1/fs)
    # raw_mag = np.abs(raw_yf) / (N/2)
    
    # search_mask = (raw_xf >= max(0, theoretical_1x - 5.0)) & (raw_xf <= theoretical_1x + 5.0)
    # if np.any(search_mask):
    #     peak_idx = np.argmax(raw_mag[search_mask])
    #     true_1x = raw_xf[search_mask][peak_idx]
    #     rpm = true_1x * 60.0
        
        # Diagnostic Plotting of the True RPM extraction
        plt.figure(figsize=(8, 4))
        plt.plot(raw_xf[search_mask], raw_mag[search_mask], color="teal", linewidth=2.0)
        plt.axvline(x=theoretical_1x, color="red", linestyle="--", label=f"Theoretical 1X ({theoretical_1x:.2f} Hz)", linewidth=2.0)
        plt.axvline(x=true_1x, color="blue", linestyle="-", label=f"True 1X Peak ({true_1x:.2f} Hz)", linewidth=2.0)
        plt.title(f"True RPM Extraction (Nominal RPM: {theoretical_rpm})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("FFT Magnitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("rpm_extraction_diagnostic.png")
        plt.close()
        
    else:
        rpm = theoretical_rpm
        
    if location is None:
        if "FE" in basename.upper():
            location = "FE"
        else:
            location = "DE"
            
    print(f"Detected Motor Load => Extracted True RPM: {rpm:.1f} (Theoretical nominal: {theoretical_rpm}, Location: {location})")
    kinematics = calculate_kinematics(rpm=rpm, location=location)
    
    # Apply Dynamic High-Pass Filter to remove unbalance/misalignment macroscopic vibrations (1X + margin)
    cutoff = 10.0 + (rpm / 60.0)
    nyq = 0.5 * fs
    sos_hp = butter(4, cutoff / nyq, btype='highpass', output='sos')
    sig_test = sosfiltfilt(sos_hp, sig_test)
    
    plot_time_series(sig_test, fs, image_path="raw_time_series.png", title="High-Pass Filtered Time Series")
    
    print("Pre-computing Advanced Signals (CPW, SK)...")
    sig_test_cpw = run_cepstrum_prewhitening(sig_test)
    sig_test_sk, opt_fc_test, sk_path_test = run_spectral_kurtosis_and_filter(sig_test, fs, "sk_test.png")
    
    fault_order = [("BSF", "BALL"), ("BPFO", "OUTER_RACE"), ("BPFI", "INNER_RACE")]
    
    def extract_confidence(diag_text):
        match = re.search(r"SCORE:\s*(-?\d+)", diag_text.upper())
        return int(match.group(1)) if match else -1
        
    threshold = 3
    
    print("\n=== EXECUTING PHASE 1: SQUARED ENVELOPE SWEEP ===")
    
    # Compute magnitude for Phase 1 Master Plot
    sig_detrend = detrend(sig_test)
    analytic_signal = hilbert(sig_detrend)
    amplitude_envelope = np.abs(analytic_signal)
    sq_env = amplitude_envelope**2
    N_1 = len(sq_env)
    yf_1 = rfft(sq_env - np.mean(sq_env))
    xf_1 = rfftfreq(N_1, 1/fs)
    mag_1 = np.abs(yf_1) / (N_1/2)
    plot_master_envelope(xf_1, mag_1, kinematics, "ph1_master_overview.png", "Phase 1: Master Envelope Overview (0-500Hz)")
    
    ph1_scores = {}
    ph1_diags = {}
    for target_fault, formal_name in fault_order:
        b64_imgs, ctx = run_envelope_analysis(sig_test, fs, kinematics, target_fault=target_fault, image_path=f"ph1_{target_fault}.png", phase_title=f"Phase 1: Squared Envelope ({target_fault})")
        
        target_score = 0
        comb_diags = []
        for b64_img, img_path, harmonic in b64_imgs:
            diagnosis = evaluate_single_harmonic_with_llm(b64_img, img_path, ctx, target_fault, harmonic, "Phase 1")
            sub_score = extract_confidence(diagnosis)
            target_score += sub_score
            comb_diags.append(f"[{harmonic}X SCORE: {sub_score}]\n{diagnosis}")
            print(f"--- AI DIAGNOSIS (PHASE 1 / {target_fault} / {harmonic}x) [SCORE: {sub_score}] ---")

        ph1_scores[formal_name] = target_score
        ph1_diags[formal_name] = "\n".join(comb_diags)
        print(f"\n========== FINAL DIAGNOSIS (PHASE 1 / {target_fault}) [TOTAL SCORE: {target_score}/3] ==========\n")
        
    best_ph1_max = max(ph1_scores.values())
    tied_ph1 = [k for k, v in ph1_scores.items() if v == best_ph1_max]
    
    if phase_1_only:
        if best_ph1_max >= 2:
            return f"{','.join(tied_ph1)}:::[Phase 1 Only] Peak score: {best_ph1_max}/3."
        else:
            return f"HEALTHY:::No significant faults found in Phase 1. Peak score was {best_ph1_max}/3."
            
    if best_ph1_max >= threshold:
        return f"{','.join(tied_ph1)}:::[Phase 1] Tied elements threshold cleared."
            
    print(f"\n*** Phase 1 max score was {best_ph1_max}/3 for {','.join(tied_ph1)} (threshold {threshold}). Escalating to PHASE 2 (CEPSTRUM PREWHITENING) ***")
    
    # Compute magnitude for Phase 2 Master Plot
    sig_detrend_c = detrend(sig_test_cpw)
    analytic_signal_c = hilbert(sig_detrend_c)
    amp_env_c = np.abs(analytic_signal_c)
    sq_env_c = amp_env_c**2
    N_2 = len(sq_env_c)
    yf_2 = rfft(sq_env_c - np.mean(sq_env_c))
    xf_2 = rfftfreq(N_2, 1/fs)
    mag_2 = np.abs(yf_2) / (N_2/2)
    plot_master_envelope(xf_2, mag_2, kinematics, "ph2_master_overview.png", "Phase 2: Master CPW Envelope Overview (0-500Hz)")
    
    ph2_scores = {}
    ph2_diags = {}
    for target_fault, formal_name in fault_order:
        b64_imgs, ctx_c = run_envelope_analysis(sig_test_cpw, fs, kinematics, target_fault=target_fault, image_path=f"ph2_{target_fault}.png", phase_title=f"Phase 2: CPW Envelope ({target_fault})")
        
        target_score = 0
        comb_diags = []
        for b64_img, img_path, harmonic in b64_imgs:
            diag_cpw = evaluate_single_harmonic_with_llm(b64_img, img_path, ctx_c, target_fault, harmonic, "Phase 2")
            sub_score = extract_confidence(diag_cpw)
            target_score += sub_score
            comb_diags.append(f"[{harmonic}X SCORE: {sub_score}]\n{diag_cpw}")
            print(f"--- AI DIAGNOSIS (PHASE 2 / {target_fault} / {harmonic}x) [SCORE: {sub_score}] ---")

        ph2_scores[formal_name] = target_score
        ph2_diags[formal_name] = "\n".join(comb_diags)
        print(f"\n========== FINAL DIAGNOSIS (PHASE 2 / {target_fault}) [TOTAL SCORE: {target_score}/3] ==========\n")
        
    best_ph2_max = max(ph2_scores.values())
    tied_ph2 = [k for k, v in ph2_scores.items() if v == best_ph2_max]
    if best_ph2_max >= threshold:
        return f"{','.join(tied_ph2)}:::[Phase 2] Tied elements threshold cleared."
            
    print(f"\n*** Phase 2 max score was {best_ph2_max}/3 for {','.join(tied_ph2)} (threshold {threshold}). Escalating to PHASE 3 (SPECTRAL KURTOSIS) ***")
    
    # Compute magnitude for Phase 3 Master Plot
    sig_detrend_s = detrend(sig_test_sk)
    analytic_signal_s = hilbert(sig_detrend_s)
    amp_env_s = np.abs(analytic_signal_s)
    sq_env_s = amp_env_s**2
    N_3 = len(sq_env_s)
    yf_3 = rfft(sq_env_s - np.mean(sq_env_s))
    xf_3 = rfftfreq(N_3, 1/fs)
    mag_3 = np.abs(yf_3) / (N_3/2)
    plot_master_envelope(xf_3, mag_3, kinematics, "ph3_master_overview.png", "Phase 3: Master SK Envelope Overview (0-500Hz)")
    
    ph3_scores = {}
    ph3_diags = {}
    for target_fault, formal_name in fault_order:
        b64_imgs, ctx_s = run_envelope_analysis(sig_test_sk, fs, kinematics, target_fault=target_fault, image_path=f"ph3_{target_fault}.png", phase_title=f"Phase 3: SK Envelope ({target_fault})")
        
        target_score = 0
        comb_diags = []
        for b64_img, img_path, harmonic in b64_imgs:
            diag_sk = evaluate_single_harmonic_with_llm(b64_img, img_path, ctx_s, target_fault, harmonic, "Phase 3")
            sub_score = extract_confidence(diag_sk)
            target_score += sub_score
            comb_diags.append(f"[{harmonic}X SCORE: {sub_score}]\n{diag_sk}")
            print(f"--- AI DIAGNOSIS (PHASE 3 / {target_fault} / {harmonic}x) [SCORE: {sub_score}] ---")

        ph3_scores[formal_name] = target_score
        ph3_diags[formal_name] = "\n".join(comb_diags)
        print(f"\n========== FINAL DIAGNOSIS (PHASE 3 / {target_fault}) [TOTAL SCORE: {target_score}/3] ==========\n")
        
    best_ph3_max = max(ph3_scores.values())
    tied_ph3 = [k for k, v in ph3_scores.items() if v == best_ph3_max]
    if best_ph3_max >= threshold:
        return f"{','.join(tied_ph3)}:::[Phase 3] Tied elements threshold cleared."
                    
    all_scores = {
        **{f"{k} (Ph1)": v for k, v in ph1_scores.items()},
        **{f"{k} (Ph2)": v for k, v in ph2_scores.items()},
        **{f"{k} (Ph3)": v for k, v in ph3_scores.items()}
    }
    absolute_best_max = max(all_scores.values())
    
    if absolute_best_max >= 2:
        tied_fallbacks = [k.split(" ")[0] for k, v in all_scores.items() if v == absolute_best_max]
        tied_fallbacks = list(set(tied_fallbacks)) # deduplicate
        return f"{','.join(tied_fallbacks)}:::[Absolute Best Fallback] Peak score: {absolute_best_max}/3."
        
    return f"HEALTHY:::No significant faults found across logic chain. Peak score was {absolute_best_max}/3."

if __name__ == "__main__":
    import sys
    target_mat_file_test = "../data/CWRU/RAW/12k_DE_IR_014_1.mat"
    phase_1_only = "--phase1" in sys.argv
    try:
        final_diagnosis = run_full_diagnosis_pipeline(target_mat_file_test, location="DE", phase_1_only=phase_1_only)
        predicted = parse_llm_diagnosis(final_diagnosis)
        print(f"\n==================================================")
        print(f"FINAL PREDICTED FAULT CLASS: {predicted}")
        print(f"==================================================\n")
    except Exception as e:
        print(f"Error: {e}")
