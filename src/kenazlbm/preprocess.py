import os
import glob
import re

def preprocess_directory(input_root="raw", output_root="preprocessed"):
    """
    Preprocess all .EDF/.edf files in the raw/*/* structure
    and save them to preprocessed/*/*_preprocessed.EDF
    
    Args:
        input_root (str): Root directory of raw files
        output_root (str): Root directory to save preprocessed files
    """
    pattern = os.path.join(input_root, "*", "*.[Ee][Dd][Ff]")
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"No files found with pattern {pattern}")
        return
    
    for infile in all_files:
        # Extract subject folder
        subject_dir = os.path.basename(os.path.dirname(infile))
        filename, ext = os.path.splitext(os.path.basename(infile))
        outfile = os.path.join(output_root, subject_dir, f"{filename}_preprocessed.EDF")
        
        print(f"Preprocessing input file: {infile}")
        # Replace this with real preprocessing logic
        preprocessed_data = f"Preprocessed content of {infile}"
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        
        # Save the preprocessed data
        print(f"Saving preprocessed data to: {outfile}")
        with open(outfile, 'w') as f:
            f.write(preprocessed_data)


def validate_directory_structure(input_root="raw"):
    """
    Validates that the directory structure is raw/*/*.[edf|EDF]
    and that filenames follow the expected format:
    <subjectID>_MMDDYYYY_HHMMSSdd.EDF
    where <subjectID> is any alphanumeric string.
    """
    pattern = os.path.join(input_root, "*", "*.[Ee][Dd][Ff]")
    all_files = glob.glob(pattern)
    
    # Regex: alphanumeric subject ID, underscore, 8 digits MMDDYYYY, underscore, 8 digits HHMMSS, 2 digits deciseconds
    filename_regex = re.compile(r"^[A-Za-z0-9]+_\d{8}_\d{8}\d{2}\.EDF$", re.IGNORECASE)
    invalid_files = []
    
    for f in all_files:
        basename = os.path.basename(f)
        if not filename_regex.match(basename):
            invalid_files.append(f)
    
    if invalid_files:
        print("The following files have invalid names:")
        for f in invalid_files:
            print(f"  {f}")
        return False
    
    print(f"All {len(all_files)} files have valid names and directory structure.")
    return True


# Example usage:
# validate_directory_structure("raw")
# preprocess_directory("raw", "preprocessed")
