# Save this as e.g., prepare_training_text.py
import json
import os

# Adjust these paths as necessary
input_json_path = './line_labels.json' # e.g., '/home/jupyter/Nusurvivors/linelabel.json'
output_text_path = os.path.expanduser('~/tesseract_custom_train/data/mycustomnusu/mycustomnusu.training_text')

all_text = []
with open(input_json_path, 'r', encoding='utf-8') as f:
    # Assuming linelabel.json is a list of dictionaries, as shown in image_63890e.png
    data = json.load(f)

for entry in data:
    all_text.append(entry['text'])

# Ensure the directory for the output text file exists
os.makedirs(os.path.dirname(output_text_path), exist_ok=True)

# Join all lines with a newline character and write to file
with open(output_text_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(all_text) + '\n') # Add a final newline for safety
print(f"Combined ground truth text written to {output_text_path}")