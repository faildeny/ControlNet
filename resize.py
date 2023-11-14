import os
import cv2
from tqdm import tqdm

directory = "training/sa_1ch_debug"
subdirs = ["source", "target"]

target_size = 512

counter = 0

for image_dir in subdirs:
    path = os.path.join(directory, image_dir)
    for filename in tqdm(os.listdir(path)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(path, filename))
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(path, filename), img)
            counter += 1

print("Resized ", counter, " images.")