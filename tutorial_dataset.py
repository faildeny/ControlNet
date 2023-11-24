import json
import cv2
import numpy as np

from torch.utils.data import Dataset


# dataset_path = "./training/sa_1ch_debug/" 
# dataset_path = "./training/stacked_EDES_resized_512/"
dataset_path = "./training/stacked_EDES_resized_128/"

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open(dataset_path+'prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        source = cv2.imread(dataset_path + source_filename)
        target = cv2.imread(dataset_path + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # # Resize source images.
        # source = cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA)
        # target = cv2.resize(target, (size, size), interpolation=cv2.INTER_AREA)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, filename=target_filename)

