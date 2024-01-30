import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = []
        self.dataset_path = dataset_path
        with open(dataset_path+'prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        print("Loaded ", len(self.data), " samples.")
        self.sample_weights = self.calculate_sample_weights()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        source = cv2.imread(self.dataset_path + source_filename)
        target = cv2.imread(self.dataset_path + target_filename)

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
    
    def get_samples_list(self):
        return self.data
    
    def calculate_sample_weights(self):
        class_sizes = {}
        sample_weights = []
        for item in self.data:
            if item['prompt'] not in class_sizes:
                class_sizes[item['prompt']] = 0
            class_sizes[item['prompt']] += 1

        # print("Class sizes: ", class_sizes)
            
        # Add weights for each sample.
        for item in self.data:
            item['sample_weight'] = 1.0 / class_sizes[item['prompt']]
            sample_weights.append(item['sample_weight'])
        
        return sample_weights
