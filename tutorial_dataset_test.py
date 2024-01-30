from tutorial_dataset import MyDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
batch_size = 2

dataset_path = "./training/stacked_EDES_fold_0_prev_0_01_resized_512/"

dataset = MyDataset(dataset_path)
print(len(dataset))

item = dataset[12]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

sampler = WeightedRandomSampler(dataset.sample_weights, len(dataset))
sampler = None
dataloader = DataLoader(dataset, num_workers=20, batch_size=batch_size, sampler=sampler)

for batch in dataloader:
    print(batch['txt'])
    break
