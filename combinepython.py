import os

# Define the target directory path
target_path = "/home/anhkhoadoannguyen/Documents/ARC AGI Sandbox/games files"
output_file = os.path.join(target_path, "combined_code.md")

# Get list of all .py files in that specific directory
try:
    py_files = [f for f in os.listdir(target_path) if f.endswith('.py')]

    if not py_files:
        print(f"No Python files found in: {target_path}")
    else:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write("# Combined Project Code\n\n")
            
            for filename in py_files:
                # Create the full path to read each file
                file_path = os.path.join(target_path, filename)
                
                outfile.write(f"## {filename}\n")
                outfile.write("```python\n")
                
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    
                outfile.write("\n```\n\n")

        print(f"Successfully combined {len(py_files)} files into {output_file}")

except FileNotFoundError:
    print(f"Error: The system cannot find the path specified: {target_path}")
