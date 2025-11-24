import os

import pandas as pd


def clean_dataset():
    base_dir = os.getcwd()

    input_filename = "dataset_train_raw.csv"
    output_filename = "dataset_train_clean.csv"
    dataset_folder = "dataset_train"

    input_csv_path = os.path.join(base_dir, input_filename)
    output_csv_path = os.path.join(base_dir, output_filename)
    dataset_root_path = os.path.join(base_dir, dataset_folder)

    if not os.path.exists(input_csv_path):
        print(f"Error: Could not find input CSV at {input_csv_path}")
        print(
            f"Make sure you are running the script from the folder containing '{input_filename}'"
        )
        return

    if not os.path.exists(dataset_root_path):
        print(f"Error: Could not find the folder '{dataset_folder}' in {base_dir}")
        return

    print(f"Opening {input_filename}...")
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    initial_count = len(df)
    print(f"Total rows loaded: {initial_count}")

    # Remove unfinished rows
    if "finished" in df.columns:
        df["finished_str"] = df["finished"].astype(str).str.lower().str.strip()
        df_filtered_finished = df[df["finished_str"] != "false"].copy()

        df_filtered_finished.drop(columns=["finished_str"], inplace=True)

        count_after_finished = len(df_filtered_finished)
        dropped_finished = initial_count - count_after_finished
        print(f"Removed {dropped_finished} rows where 'finished' = false.")
    else:
        print("Warning: Column 'finished' not found in CSV. Skipping this step.")
        df_filtered_finished = df

    # Index all files in subfolders
    print("Scanning directory tree for actual files (this may take a moment)...")
    available_files = set()

    for root, dirs, files in os.walk(dataset_root_path):
        for file in files:
            available_files.add(file)

    print(
        f"Found {len(available_files)} files in '{dataset_folder}' and its subfolders."
    )

    # Filter rows based on files
    valid_indices = []
    missing_files_count = 0

    for index, row in df_filtered_finished.iterrows():
        fname = row.get("filename")

        if pd.isna(fname):
            missing_files_count += 1
            continue

        base_filename = os.path.basename(fname)

        if base_filename in available_files:
            valid_indices.append(index)
        else:
            missing_files_count += 1
            # print(f"File missing: {fname}")

    final_df = df_filtered_finished.loc[valid_indices].copy()
    final_df = final_df.sort_values(by="filename", ascending=True)

    print(f"Removed {missing_files_count} rows where the file was not found.")
    print(f"Final row count: {len(final_df)}")

    # Save to new location
    try:
        final_df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved cleaned data to: {output_filename}")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    clean_dataset()
