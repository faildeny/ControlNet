import os
import cv2
from tqdm import tqdm

directory = "training/stacked_EDES"
subdirs = ["source", "target"]


target_size = 128

counter = 0

for image_dir in subdirs:
    path = os.path.join(directory, image_dir)
    output_path = os.path.join(directory + "_resized_" + target_size, image_dir)
    os.makedirs(output_path, exist_ok=True)
    for filename in tqdm(os.listdir(path)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(path, filename))
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(output_path, filename), img)
            counter += 1

prompt_path = os.path.join(directory + "_resized_" + target_size, "prompt.json")
os.copy(prompt_path, os.path.join(directory + "_resized_" + target_size, "prompt.json"))

print("Resized ", counter, " images.")