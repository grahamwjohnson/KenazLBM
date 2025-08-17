import os
import glob
import re

def preprocess_directory(in_dir="raw", out_dir=None):
    """
    Preprocess all .EDF/.edf files in the raw/*/* structure
    and save them to preprocessed/*/*_pp.pkl
    
    Args:
        in_dir (str): Root directory of raw files
        out_dir (str): Root directory to save preprocessed files
    """

    if out_dir is None: out_dir = in_dir

    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Find all EDF files in the input directory
    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

    # Use glob to find all EDF files in the structure
    pattern = os.path.join(in_dir, "*", "*.[Ee][Dd][Ff]")
    all_files = glob.glob(pattern)
    
    for infile in all_files:
        # Extract subject folder
        subject_dir = os.path.basename(os.path.dirname(infile))
        filename, ext = os.path.splitext(os.path.basename(infile))
        outfile = os.path.join(out_dir, subject_dir, f"{filename}_pp.pkl")
        
        print(f"Preprocessing input file: {infile}")
        
        


        # Replace this with real preprocessing logic
        # TODO
        preprocessed_data = f"Preprocessed content of {infile}"
        



        # Ensure the output directory exists
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        
        # Save the preprocessed data
        print(f"Saving preprocessed data to: {outfile}")
        with open(outfile, 'w') as f:
            f.write(preprocessed_data)


# Example usage:
# validate_directory_structure("raw")
# preprocess_directory("raw", "preprocessed")
