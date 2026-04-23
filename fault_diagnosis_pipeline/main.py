import argparse
from data_loader import generate_vibration_data
from signal_processing import compare_to_baseline
from llm_analyzer import diagnose_fault_with_llm

def main():
    parser = argparse.ArgumentParser(description="Intelligent Fault Diagnosis Pipeline")
    parser.add_argument("--condition", type=str, default="imbalance", 
                        choices=["healthy", "imbalance", "misalignment", "bearing", "IR", "OR", "B"],
                        help="The simulated condition to test against the healthy baseline.")
    parser.add_argument("--rpm", type=float, default=1800.0, help="Rotational speed of the machine in RPM")
    parser.add_argument("--model", type=str, default="gemma4", help="Ollama model to use for the LLM analysis")
    parser.add_argument("--dataset", type=str, default="cwru", choices=["dummy", "cwru"], help="Dataset to use for data loading.")
    
    args = parser.parse_args()

    print(f"--- Running Intelligent Fault Diagnosis Pipeline ---")
    
    if args.dataset == "dummy":
        print(f"Simulating Machine RPM: {args.rpm}")
        print(f"1. Measure: Generating healthy baseline and '{args.condition}' mock signals...")
        t_base, sig_base = generate_vibration_data(condition="healthy", rpm=args.rpm)
        t_test, sig_test = generate_vibration_data(condition=args.condition, rpm=args.rpm)
    elif args.dataset == "cwru":
        print(f"1. Measure: Loading CWRU real-world base and test signals...")
        from cwru_data_loader import get_cwru_data
        t_base, sig_base, rpm_base = get_cwru_data(condition="healthy", torque=0)
        t_test, sig_test, rpm_test = get_cwru_data(condition=args.condition, torque=0)
        args.rpm = rpm_test # Update args to true RPM for LLM context
        print(f"Dataset Actual Machine RPM: {args.rpm} (Torque 0)")

    
    print(f"2. Transform & 3. Compare: Running FFT, extracting peaks, comparing against baseline...")
    base_peaks, test_peaks, diffs = compare_to_baseline(t_base, sig_base, t_test, sig_test)
    
    if not diffs:
        print(">> No significant algorithmic anomalies detected. Still passing to Gemma for visual assessment.")
    else:
        print(f">> Detected Anomalies: {diffs}")
    
    print("4. Transform: Creating FFT image for LLM interpretation...")
    image_path = "fft_plot.png"
    plt_instance = None
    try:
        from signal_processing import compute_fft
        import matplotlib.pyplot as plt
        
        f_base, m_base = compute_fft(t_base, sig_base)
        f_test, m_test = compute_fft(t_test, sig_test)
        
        plt.figure(figsize=(10, 6))
        plt.plot(f_base, m_base, label="Healthy Baseline", color="blue", linestyle="-", linewidth=2, alpha=0.6)
        plt.plot(f_test, m_test, label="Test Signal", color="red", linestyle="--", linewidth=1.5, alpha=0.8)
        
        for peak in diffs:
            plt.scatter(peak['freq'], peak['mag'], color='black', zorder=5)
            plt.text(peak['freq'] + 2, peak['mag'] + 0.1, f"{peak['freq']}Hz", fontsize=9)
            
        plt.title("FFT Spectrum Comparison: Healthy Baseline vs Test Signal")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        
        max_freq = max([p['freq'] for p in diffs]) + 50 if diffs else 200
        plt.xlim(0, max(200, max_freq))
        
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(image_path)
        plt_instance = plt
    except ImportError:
        print("Warning: matplotlib must be installed to create the plot image.")
        image_path = None
    
    print(f"5. Identify: Sending FFT image to LLM ({args.model}) for visual diagnostic reasoning...\n")
    diagnosis = diagnose_fault_with_llm(diffs, rpm=args.rpm, model_name=args.model, image_path=image_path)
    
    print("========== AI EXPERT DIAGNOSIS ==========\n")
    print(diagnosis)
    print("\n=========================================")
    
    if plt_instance:
        print("6. Display: Showing generated FFT Comparison Plot (close window to exit)...")
        plt_instance.show()

if __name__ == "__main__":
    main()
