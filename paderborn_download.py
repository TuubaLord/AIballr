import os
import shutil
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import patoolib
from scipy.io import loadmat


def download(url: str, filepath: str):
    try:
        urlretrieve(url, filepath)
    except Exception as e:
        if Path(filepath).exists():
            Path(filepath).unlink()
        print(f"Download failed for {url}: {e}")
        raise


def download_paderborn_dataset(download_dir):
    """Download all Paderborn dataset files to download_dir"""

    print("Downloading the Paderborn dataset...")
    print()

    base_url = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"
    files = [
        "K001.rar",
        "K002.rar",
        "K003.rar",
        "K004.rar",
        "K005.rar",
        "K006.rar",
        "KA01.rar",
        "KA03.rar",
        # "KA04.rar",
        "KA05.rar",
        "KA06.rar",
        "KA07.rar",
        "KA08.rar",
        "KA09.rar",
        # "KA15.rar",
        # "KA16.rar",
        # "KA22.rar",
        # "KA30.rar",
        # "KB23.rar",
        # "KB24.rar",
        # "KB27.rar",
        "KI01.rar",
        "KI03.rar",
        # "KI04.rar",
        "KI05.rar",
        "KI07.rar",
        "KI08.rar",
        # "KI14.rar",
        # "KI16.rar",
        # "KI17.rar",
        # "KI18.rar",
        # "KI21.rar",
    ]

    total = len(files)
    for i, file in enumerate(files):
        file_url = base_url + file
        print("Downloading file {} / {}".format(i + 1, total))

        if (Path(download_dir) / file).exists():
            print(f"File {file} already exists. Skipping download.")
            continue
        download(file_url, download_dir / file)

    print()
    print("Download complete.")


def process_dataset(download_dir, interim_dir):
    """Process .mat files from download_dir and save parquet chunks to interim_dir"""

    nominal_speed_map = {
        "N09": 900,
        "N15": 1500,
    }

    load_torque_map = {
        "M01": 0.1,
        "M07": 0.7,
    }

    radial_force_map = {
        "F04": 0.4,
        "F10": 1.0,
    }

    print("Starting data processing...")
    print()

    # Loop directories #

    dirs = download_dir.glob("**/*.rar")
    for d in dirs:
        print(f"Processing {d}...")
        saved_data = {
            "bearing": [],
            "measurement_num": [],
            "nominal_speed": [],
            "load": [],
            "radial_force": [],
            "speed": [],
            "vibration": [],
        }

        if (interim_dir / f"{d.stem}.parquet").exists():
            print(
                f"Parquet file for {d.stem} already exists in interim directory. Skipping processing."
            )
            continue

        # Check if the extracted directory already exists before extracting
        delete_extracted_dir = True
        extracted_dir = download_dir / d.stem
        if not extracted_dir.exists():
            try:
                patoolib.extract_archive(
                    str(d.absolute()), outdir=download_dir, verbosity=-1
                )
            except Exception as e:
                print(f"\033[91mError:\033[0m Error occurred while extracting {d}: {e}")
                print(
                    "Could not extract the archive with Patool. If this was the first .rar file, the problem is most likely missing compatible unarchiver software (e.g. unrar). Please install one or unarchive manually."
                )
        else:
            delete_extracted_dir = False
            print(f"Directory {extracted_dir} already exists. Skipping extraction.")

        # Loop files #

        files = download_dir.glob("**/*.mat")
        for f in files:
            name_parts = f.stem.split("_")

            # Load data
            # NOTE: Some .mat files in the dataset are corrupted.
            # Specifically: `KA08\N15_M01_F10_KA08_2.mat`
            try:
                data = loadmat(f)
            except Exception as e:
                print(f"\033[93mWarning:\033[0m Error occurred while loading {f}: {e}")
                continue

            # Save specs
            # FIXME: Add fault
            saved_data["bearing"].append(name_parts[-2])
            saved_data["measurement_num"].append(name_parts[-1])
            saved_data["nominal_speed"].append(nominal_speed_map[name_parts[0]])
            saved_data["load"].append(load_torque_map[name_parts[1]])
            saved_data["radial_force"].append(radial_force_map[name_parts[2]])

            # Find the relevant signals by name
            signal_list = data[f.stem][0, 0]["Y"].squeeze()
            for signal in signal_list:
                name = signal[0][0]
                if name not in ["speed", "vibration_1"]:
                    continue

                # Save only the relevant signals, and convert to float32 to save memory
                if name == "vibration_1":
                    values = signal[2][0].flatten()
                    saved_data["vibration"].append(values.astype(np.float32))
                elif name == "speed":
                    values = signal[2][0].flatten()
                    saved_data["speed"].append(values.astype(np.float32))

        df = pd.DataFrame(saved_data)
        df.to_parquet(
            interim_dir / f"{d.stem}.parquet", compression="zstd", index=False
        )

        # Remove the directory if it was created by this script (i.e. it didn't exist beforehand)
        if delete_extracted_dir:
            shutil.rmtree(download_dir / d.stem)

    print("Everything processed and saved to interim directory.")


if __name__ == "__main__":
    # Get current location of the script to determine where to download and save data
    try:
        # Works in .py script
        cur_dir = Path(__file__).resolve().parent
    except NameError:
        # Fallback for notebooks where __file__ is undefined
        cur_dir = Path(os.getcwd())

    # Define locations

    project_root = cur_dir

    download_dir = project_root / "data" / "Paderborn" / "RAW"
    download_dir.mkdir(parents=True, exist_ok=True)

    interim_dir = project_root / "data" / "Paderborn" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    download_paderborn_dataset(download_dir)

    # Process dataset
    process_dataset(download_dir, interim_dir)
