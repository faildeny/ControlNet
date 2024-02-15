import json
import os
dataset_path = "training/stacked_EDES_fold_0_prev_0_01"
new_dataset_path = "training/stacked_EDES_fold_0_prev_0_01_renamed"
prompt_name = "prompt.json"
subdirs = ["target", "source"]
# open prompt file
samples = []  
with open(os.path.join(dataset_path, prompt_name), "r") as f:
    for line in f:
        samples.append(json.loads(line))

# Create new directory
os.makedirs(new_dataset_path, exist_ok=True)
# Create subdirs
for subdir in subdirs:
    os.makedirs(os.path.join(new_dataset_path, subdir), exist_ok=True)
# Create new prompt file

# Change file paths for each sample and add prompt to the file name
for sample in samples:
    prompt = sample["prompt"]
    target_path = sample["target"]
    source_path = sample["source"]
    paths = [target_path, source_path]
    # Remove special characters from prompt
    prompt = prompt.replace(", ", "_").replace(" ", "_")
    for path in paths:
        old_full_path = os.path.join(dataset_path, path)
        new_path = path.split(".")[0] + "_" + prompt + "." + path.split(".")[1]
        new_full_path = os.path.join(new_dataset_path, new_path)
        # Replace path in target and source fields
        if path == target_path:
            sample["target"] = new_path
        else:
            sample["source"] = new_path
        os.rename(old_full_path, new_full_path)

# Write new prompt file
new_prompt_path = os.path.join(new_dataset_path, prompt_name)
with open(new_prompt_path, "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")

print("Done")
