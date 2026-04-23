import pandas as pd
import numpy as np
from pathlib import Path

def get_cwru_data(condition="IR", torque=0, sampling_frequency=12000, duration=1.0):
    """
    Load real CWRU data for testing.
    
    Args:
        condition (str): "healthy" (normal), "IR" (inner race), "OR" (outer race), or "B" (rolling element / ball).
        torque (int): 0, 1, 2, or 3.
        sampling_frequency (int): 12000 or 48000.
        duration (float): How many seconds of data to load.
        
    Returns:
        t (np.ndarray): time vector
        sig (np.ndarray): vibration data
        rpm (float): rotational speed
    """
    # Assuming this is run from fault_diagnosis_pipeline and data was downloaded from parent dir
    parquet_path = Path(__file__).parent.parent / "data" / "CWRU" / "CWRU_downloaded.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"CWRU dataset not found at {parquet_path}. Please run the download script.")
    
    fault_type_val = "normal" if condition.lower() == "healthy" else condition
    
    if fault_type_val == "normal":
        # The CWRU script hardcoded normal to 48kHz
        sr_val = 48
        sampling_frequency = 48000
    else:
        sr_val = sampling_frequency // 1000
    
    print(f"[*] Accessing CWRU dataset (Parquet) for '{fault_type_val}' condition at {torque}HP load, {sr_val}kHz...")
    
    # Read subset from parquet
    df = pd.read_parquet(
        parquet_path,
        columns=["measurement_id", "sample_index", "measurement", "fault location", "torque", "sampling rate", "measurement location"],
        filters=[
            ("fault location", "==", fault_type_val if fault_type_val == "normal" else fault_type_val),
            ("torque", "==", torque),
            ("sampling rate", "==", sr_val),
            ("measurement location", "==", "DE") # We use Drive End sensors for the signal
        ] # wait, in cwru_download.py, fault_location is "IR", "OR", etc. Sometimes fault type is.
    )
    
    # Wait, looking closely at cwru_download.py:
    # fault_location = measurement_specs[1] 
    # fault_type = measurement_specs[2].split("-")[0]
    # In cwru_download.py: "48k_DE_OR-C_007_0.mat", split("_") => ['48k', 'DE', 'OR-C', '007', '0']
    # If len=5: [0] = 48k => sr. [1] = DE => location. [2] = OR-C => fault_type. [3]= depth. [4] = torque
    # So `fault type` actually stores "OR", "IR", "B". And `fault location` stores "DE", "FE". Wait, cwru_download.py:
    # fault_location = measurement_specs[1] (e.g. DE)
    # fault_type = measurement_specs[2].split("-")[0] (e.g. OR)
    
    df = pd.read_parquet(
        parquet_path,
        columns=["measurement_id", "sample_index", "measurement", "fault type", "torque", "sampling rate", "measurement location"],
        filters=[
            ("fault type", "==", fault_type_val),
            ("torque", "==", torque),
            ("sampling rate", "==", sr_val),
            ("measurement location", "==", "DE")
        ]
    )
    
    if df.empty:
        raise ValueError(f"No CWRU data found for condition '{condition}', torque '{torque}', sampling_rate '{sampling_frequency}'")
        
    # Get the first measurement ID available
    first_id = df['measurement_id'].iloc[0]
    
    # Filter only that measurement ID to get a contiguous signal
    signal_df = df[df['measurement_id'] == first_id].sort_values("sample_index")
    sig = signal_df['measurement'].values
    
    # Limit to duration
    num_samples = int(duration * sampling_frequency)
    if len(sig) > num_samples:
        sig = sig[:num_samples]
        
    t = np.arange(0, len(sig)) / sampling_frequency
    
    # Roughly map torque to CWRU RPM speeds
    rpm_map = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}
    rpm = rpm_map.get(torque, 1797.0)
    
    return t, sig, rpm

if __name__ == "__main__":
    t, sig, rpm = get_cwru_data(condition="IR")
    print(f"Successfully loaded IR signal of length {len(sig)}")
