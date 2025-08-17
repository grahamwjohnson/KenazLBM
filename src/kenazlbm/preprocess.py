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


import os
import glob
import re

def validate_directory_structure(input_root="raw", file_pattern="*.edf"): 
    """
    Validates that the directory structure is raw/*/<file_pattern>
    and that filenames follow the expected format:
    <subjectID>_MMDDYYYY_HHMMSSdd.<ext>
    where <subjectID> is any alphanumeric string.
    
    Args:
        input_root (str): Root directory (default "raw")
        file_pattern (str): Glob pattern for files (default "*.edf")
    """
    pattern = os.path.join(input_root, "*", file_pattern)
    all_files = glob.glob(pattern)

    # Regex for filenames: <subjectID>_<MMDDYYYY>_<HHMMSSdd>.<ext>
    #   - subjectID = letters/numbers
    #   - MMDDYYYY = 8 digits
    #   - HHMMSS = 6 digits
    #   - dd = 2 digits (deciseconds)
    #   - extension: whatever is in file_pattern, handled case-insensitive
    ext = os.path.splitext(file_pattern)[-1].lstrip("*.")
    filename_regex = re.compile(
        rf"^[A-Za-z0-9]+_\d{{8}}_\d{{8}}\d{{2}}\.{ext}$", re.IGNORECASE
    )

    invalid_files = []
    
    if not all_files:
        print(f"No files found with pattern {pattern}")
        return False

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
