import re

def extract_experiment_ids(file_path):
    with open(file_path, 'r') as f:
        log_file = f.read()
    experiment_ids = re.findall(r'Running experiment with ID (\S+)', log_file)
    return experiment_ids

# Example usage
file_path = './scratchpad/log.txt'
print(f"Extracting IDs from {file_path}.\n")
experiment_ids = '\", \"'.join(extract_experiment_ids(file_path))
print('[\"'+experiment_ids+'\"]')