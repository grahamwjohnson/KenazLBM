import os
import glob
import re

import os
import glob

def preprocess_directory(in_dir="raw", out_dir=None):
    """
    Preprocess all .EDF/.edf files in the raw/*/* structure
    and save them to preprocessed/*/*_pp.pkl per subject directory.

    Args:
        in_dir (str): Root directory of raw files
        out_dir (str, optional): Root directory to save preprocessed files.
                                 If None, defaults to the input directory.
    """
    if out_dir is None:
        out_dir = in_dir

    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # Find all subject directories under input root
    subject_dirs = [d for d in glob.glob(os.path.join(in_dir, "*")) if os.path.isdir(d)]

    if not subject_dirs:
        print(f"No subject directories found under {in_dir}.")
        return

    # Go through each subject directory one at a time because need to equalize at subject level
    print(f"Found {len(subject_dirs)} subject directories under {in_dir}.")
    for subj_path in subject_dirs:
        subject_id = os.path.basename(subj_path)
        print(f"\nProcessing subject: {subject_id}")

        # Find all EDF files in this subject directory
        edf_files = glob.glob(os.path.join(subj_path, "*.[Ee][Dd][Ff]"))

        if not edf_files:
            print(f"  No EDF files found for subject {subject_id}, skipping.")
            continue

        for infile in edf_files:
            filename, ext = os.path.splitext(os.path.basename(infile))
            outfile = os.path.join(out_dir, subject_id, f"{filename}_pp.pkl")

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(outfile), exist_ok=True)

            print(f"  Preprocessing input file: {infile}")

            # TODO: Replace this with actual preprocessing logic
            preprocessed_data = f"Preprocessed content of {infile}"

            # Save preprocessed data
            print(f"  Saving preprocessed data to: {outfile}")
            with open(outfile, 'w') as f:
                f.write(preprocessed_data)

    print("\nAll subjects processed successfully.")



# Example usage:
# validate_directory_structure("raw")
# preprocess_directory("raw", "preprocessed")
