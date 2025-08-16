import os

def preprocess_file(input_file, output_file=None):
    """
    Preprocess the input file and save the output.
    
    If no output_file is provided, saves to the same directory as the input file.
    If output_file is provided, saves to the specified path.
    
    Args:
        input_file (str): Path to the input file to preprocess.
        output_file (str, optional): Path to save the preprocessed output. Defaults to None.
    """
    print(f"Preprocessing input file: {input_file}")
    
    # Example preprocessing logic
    preprocessed_data = f"Preprocessed content of {input_file}"  # replace with real logic
    
    # Determine output path
    if output_file:
        save_path = output_file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # ensure directory exists
    else:
        # Save in same directory as input, keep same filename
        input_dir = os.path.dirname(input_file)
        input_name = os.path.basename(input_file)
        save_path = os.path.join(input_dir, input_name)
    
    # Save the preprocessed data
    print(f"Saving preprocessed data to: {save_path}")
    with open(save_path, 'w') as f:
        f.write(preprocessed_data)


