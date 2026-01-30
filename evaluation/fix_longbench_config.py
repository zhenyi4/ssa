import os
import re

# Path relative to where you run the script
target_dir = "../lm-evaluation-harness/lm_eval/tasks/longbench/"
target_path = os.path.abspath(target_dir)

if not os.path.exists(target_path):
    print(f"Error: Directory not found at {target_path}")
    exit(1)

print(f"Processing YAML files in: {target_path}\n")

# Regex to capture the key and the content inside single quotes
# Group 1: key (doc_to_text or doc_to_target)
# Group 2: content (everything between the first and last single quote)
# Note: This regex assumes the value is on a single line (standard for these config files)
quote_pattern = re.compile(r"^(doc_to_text|doc_to_target):\s*'(.*)'\s*$")

# Regex for do_sample
do_sample_pattern = re.compile(r"do_sample:\s*True")

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    changes = 0

    for line in lines:
        original_line = line
        
        # 1. Fix do_sample: True -> False
        if do_sample_pattern.search(line):
            line = do_sample_pattern.sub("do_sample: False", line)

        # 2. Fix quotes: '...' -> "..." with escaping
        match = quote_pattern.match(line.strip())
        if match:
            key = match.group(1)
            content = match.group(2)
            
            # CRITICAL FIX: Escape existing double quotes inside the content
            # If the text was: Say "Hello"
            # It becomes: Say \"Hello\"
            escaped_content = content.replace('"', '\\"')
            
            # Reconstruct the line with double quotes
            # Preserves original indentation if any (though these keys are usually root)
            line = f"{key}: \"{escaped_content}\"\n"

        if line != original_line:
            changes += 1
        
        new_lines.append(line)

    if changes > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"✅ Fixed {os.path.basename(file_path)} ({changes} changes)")
    else:
        print(f"➖ Skipped {os.path.basename(file_path)} (No changes needed)")

# Run the fix
for filename in os.listdir(target_path):
    if filename.endswith(".yaml") or filename.endswith(".yml"):
        process_file(os.path.join(target_path, filename))

print("\nProcessing complete.")
