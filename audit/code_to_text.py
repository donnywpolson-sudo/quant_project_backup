import os
import re

def clean_code(content):
    """Token-efficient cleaning: remove docstrings, extra newlines, and comments."""
    # Remove docstrings (triple quotes)
    content = re.sub(r'"""[\s\S]*?"""', '', content)
    content = re.sub(r"'''[\s\S]*?'''", '', content)
    # Remove inline comments
    content = re.sub(r'#.*', '', content)
    # Remove excessive blank lines (more than 1)
    content = re.sub(r'\n\s*\n', '\n\n', content)
    return content

def git_ingest(output_dir="audit", output_filename="full_code.txt"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    full_output_path = os.path.join(output_dir, output_filename)
    target_extensions = ('.py', '.yaml')
    
    with open(full_output_path, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk('.'):
            if output_dir in root.split(os.sep):
                continue
                
            for file in files:
                if file.endswith(target_extensions):
                    file_path = os.path.join(root, file)
                    
                    outfile.write("="*80 + "\n")
                    outfile.write(f"FILE: {file_path}\n")
                    outfile.write("="*80 + "\n\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            # Apply token-saving cleaning
                            if file.endswith('.py'):
                                content = clean_code(content)
                            outfile.write(content)
                    except Exception as e:
                        outfile.write(f"Error reading file: {e}")
                        
                    outfile.write("\n\n")

    print(f"Token-optimized ingestion complete: {full_output_path}")

if __name__ == "__main__":
    git_ingest()