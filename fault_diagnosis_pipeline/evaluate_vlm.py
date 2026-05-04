import os
import glob
import csv
import argparse
from multimodal_diagnosis import run_full_diagnosis_pipeline, parse_llm_diagnosis

def print_confusion_matrix(y_true, y_pred):
    classes_true = ["HEALTHY", "BALL", "INNER_RACE", "OUTER_RACE"]
    classes_pred = ["HEALTHY", "BALL", "INNER_RACE", "OUTER_RACE", "UNDIAGNOSABLE"]
    print("\n" + "="*70)
    print("CONFUSION MATRIX (Row=Ground Truth, Col=Predicted)")
    print("="*70)
    
    header = f"{'':<12} | " + " | ".join([f"{c[:5]:<5}" for c in classes_pred]) + " |"
    print(header)
    print("-" * len(header))
    
    for t in classes_true:
        row_str = f"{t[:12]:<12} | "
        for p in classes_pred:
            count = 0.0
            for gt, pr in zip(y_true, y_pred):
                if gt == t:
                    preds = [x.strip() for x in pr.split(',')]
                    if p in preds:
                        count += 1.0 / len(preds)
            row_str += f"{count:<5.1f} | "
        print(row_str)

def determine_ground_truth(filename):
    basename = os.path.basename(filename).lower()
    
    # CWRU Mappings
    if 'normal' in basename:
        return 'HEALTHY'
    elif 'de_b' in basename or 'fe_b' in basename:
        return 'BALL'
    elif 'de_ir' in basename or 'fe_ir' in basename:
        return 'INNER_RACE'
    elif 'de_or' in basename or 'fe_or' in basename:
        return 'OUTER_RACE'
        
    # Paderborn Mappings
    basename_no_ext = basename.replace('.parquet', '')
    if basename_no_ext.startswith('k0'):
        return 'HEALTHY'
    elif basename_no_ext in ['ka01', 'ka03', 'ka05', 'ka06', 'ka07', 'ki01', 'ki03', 'ki05', 'ki07']:
        return 'OUTER_RACE'
    elif basename_no_ext in ['ka04', 'ka08', 'ka09', 'ki04', 'ki08']:
        return 'INNER_RACE'
        
    return 'UNKNOWN'

def run_evaluation(filter_file=None, target_phase=None):
    dataset_type = 'CWRU'
    if filter_file and 'pu' in filter_file.lower():
        dataset_type = 'PU'
        data_dir = "../data/Paderborn/interim/"
    else:
        dataset_type = 'CWRU'
        data_dir = "../data/CWRU/RAW/"
        
    if filter_file:
        if not filter_file.endswith('.csv'):
            filter_file += '.csv'
        if not os.path.exists(filter_file):
            print(f"Filter file {filter_file} not found!")
            return
        with open(filter_file, "r") as f:
            filenames = [line.strip() for line in f if line.strip()]
        files = [os.path.join(data_dir, fname) for fname in filenames]
        print(f"Using {len(files)} files specified from {filter_file} (Dataset: {dataset_type}).")
    else:
        # Get all 12k DE files and normal files
        files = glob.glob(os.path.join(data_dir, "12k_DE_*.mat")) + glob.glob(os.path.join(data_dir, "normal_*.mat"))
        print(f"Found {len(files)} CWRU files to evaluate globally.")
    
    total = 0
    correct_anomaly = 0
    correct_diagnosis = 0
    
    y_true = []
    y_pred = []
    results_list = []
    
    for filepath in files:
        ground_truth = determine_ground_truth(filepath)
        if ground_truth == 'UNKNOWN':
            print(f"Skipping {filepath} (Unknown Ground Truth)")
            continue
            
        print(f"\n==================================================")
        print(f"Evaluating: {os.path.basename(filepath)} (Ground Truth: {ground_truth})")
        print(f"==================================================")
        
        try:
            if ground_truth == 'HEALTHY' and dataset_type == 'CWRU':
                print(f"Evaluating HEALTHY file {os.path.basename(filepath)} for Drive End (DE) faults...")
                out_dir_de = os.path.join("evaluate_vlm_results", f"{os.path.basename(filepath)}_DE")
                os.makedirs(out_dir_de, exist_ok=True)
                raw_diag_de = run_full_diagnosis_pipeline(filepath, location="DE", target_phase=target_phase, output_dir=out_dir_de)
                pred_de = parse_llm_diagnosis(raw_diag_de)
                with open(os.path.join(out_dir_de, "diagnosis.txt"), "w") as f:
                    f.write(f"Predicted: {pred_de}\n\nRaw Output:\n{raw_diag_de}")
                
                print(f"Evaluating HEALTHY file {os.path.basename(filepath)} for Fan End (FE) faults...")
                out_dir_fe = os.path.join("evaluate_vlm_results", f"{os.path.basename(filepath)}_FE")
                os.makedirs(out_dir_fe, exist_ok=True)
                raw_diag_fe = run_full_diagnosis_pipeline(filepath, location="FE", target_phase=target_phase, output_dir=out_dir_fe)
                pred_fe = parse_llm_diagnosis(raw_diag_fe)
                with open(os.path.join(out_dir_fe, "diagnosis.txt"), "w") as f:
                    f.write(f"Predicted: {pred_fe}\n\nRaw Output:\n{raw_diag_fe}")
                
                # Append DE
                y_true.append('HEALTHY')
                y_pred.append(pred_de)
                results_list.append({
                    "file": f"{os.path.basename(filepath)} (DE Eval)",
                    "truth": 'HEALTHY',
                    "predicted": pred_de,
                    "raw_diag_snippet": raw_diag_de.replace('\n', ' ')[:200]
                })
                total += 1
                preds_de = [x.strip() for x in pred_de.split(',')]
                pred_is_anomaly_de = all(p != 'HEALTHY' and p != 'UNDIAGNOSABLE' for p in preds_de)
                if not pred_is_anomaly_de:
                    correct_anomaly += 1
                if 'HEALTHY' in preds_de:
                    correct_diagnosis += 1.0 / len(preds_de)
                    
                # Append FE
                y_true.append('HEALTHY')
                y_pred.append(pred_fe)
                results_list.append({
                    "file": f"{os.path.basename(filepath)} (FE Eval)",
                    "truth": 'HEALTHY',
                    "predicted": pred_fe,
                    "raw_diag_snippet": raw_diag_fe.replace('\n', ' ')[:200]
                })
                total += 1
                preds_fe = [x.strip() for x in pred_fe.split(',')]
                pred_is_anomaly_fe = all(p != 'HEALTHY' and p != 'UNDIAGNOSABLE' for p in preds_fe)
                if not pred_is_anomaly_fe:
                    correct_anomaly += 1
                if 'HEALTHY' in preds_fe:
                    correct_diagnosis += 1.0 / len(preds_fe)
                    
                print(f"--> Predicted DE: {pred_de} | Predicted FE: {pred_fe} | Ground Truth: HEALTHY")
                continue
                
            else:
                out_dir = os.path.join("evaluate_vlm_results", os.path.basename(filepath))
                os.makedirs(out_dir, exist_ok=True)
                kwargs = {'target_phase': target_phase, 'output_dir': out_dir}
                if dataset_type == 'PU':
                    kwargs['location'] = 'PU'
                    kwargs['measurement_index'] = 0
                    
                raw_diag = run_full_diagnosis_pipeline(filepath, **kwargs)
                predicted = parse_llm_diagnosis(raw_diag)
                with open(os.path.join(out_dir, "diagnosis.txt"), "w") as f:
                    f.write(f"Predicted: {predicted}\n\nRaw Output:\n{raw_diag}")
                
            print(f"--> Predicted: {predicted} | Ground Truth: {ground_truth}")
            
            y_true.append(ground_truth)
            y_pred.append(predicted)
            results_list.append({
                "file": os.path.basename(filepath),
                "truth": ground_truth,
                "predicted": predicted,
                "raw_diag_snippet": raw_diag.replace('\n', ' ')[:200]
            })
            
            total += 1
            is_anomaly = ground_truth != 'HEALTHY'
            preds = [x.strip() for x in predicted.split(',')]
            pred_is_anomaly = all(p != 'HEALTHY' and p != 'UNDIAGNOSABLE' for p in preds)
            
            # Anomaly Detection (Is it healthy vs is it faulty)
            if is_anomaly == pred_is_anomaly:
                correct_anomaly += 1
                
            # Exact Specific Fault Diagnosis
            if ground_truth in preds:
                correct_diagnosis += 1.0 / len(preds)
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    if total > 0:
        print("\n\n================ FINAL EVALUATION RESULTS ================")
        print(f"Total Files Tested: {total}")
        print(f"Correct Anomaly Detections (Binary Healthy vs Faulty): {correct_anomaly}/{total} ({correct_anomaly/total*100:.1f}%)")
        print(f"Correct Specific Diagnoses (Exact Fault Match): {correct_diagnosis}/{total} ({correct_diagnosis/total*100:.1f}%)")
        print("==========================================================")
        
        print_confusion_matrix(y_true, y_pred)
        
        # Save to CSV
        csv_file = "vlm_evaluation_results.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["file", "truth", "predicted", "raw_diag_snippet"])
            writer.writeheader()
            writer.writerows(results_list)
        print(f"\nSaved file-by-file truth, prediction, and log snippet to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM against CWRU and Paderborn datasets.")
    parser.add_argument("--filter", type=str, help="Restricts the test suite to only evaluate files explicitly listed in the specified filter (e.g. filter_cwru or filter_pu).")
    parser.add_argument("--phase1", action="store_true", help="Run only Phase 1 (Squared Envelope).")
    parser.add_argument("--phase2", action="store_true", help="Run only Phase 2 (Cepstrum Prewhitening).")
    parser.add_argument("--phase3", action="store_true", help="Run only Phase 3 (Spectral Kurtosis).")
    args = parser.parse_args()
    
    target_phase = None
    if args.phase1: target_phase = 1
    elif args.phase2: target_phase = 2
    elif args.phase3: target_phase = 3
    
    run_evaluation(filter_file=args.filter, target_phase=target_phase)
