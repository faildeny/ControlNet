from share import *
import config
import os
from pathlib import Path

from tqdm import tqdm
import cv2
import einops
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


source_dataset_path = "./training/stacked_EDES_resized_128"

# model_checkpoint = './lightning_logs/version_21/checkpoints/epoch=0-step=30296.ckpt' # less than 1 epoch
model_checkpoint = 'lightning_logs/version_23/checkpoints/epoch=3-step=121187.ckpt' # 512
# model_checkpoint = 'lightning_logs/version_43/checkpoints/epoch=18-step=71971.ckpt' # 128 

# Features for random prompt generation
sex = ['Male', 'Female']
age = ['age in 50s', 'age in 60s', 'age in 70s', 'age in 80s', 'age in 90s']
bmi = ['normal BMI', 'overweight BMI', 'obese BMI']
diagnosis = ['healthy', 'heart failure']

features = [sex, age, bmi, diagnosis]

# Initialize model
model = create_model('./models/cldm_v21.yaml').cuda()
model.load_state_dict(load_state_dict(model_checkpoint, location='cpu'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

dataset = MyDataset()

# Load paths to masks with diagnosed heart failure
def get_masks_list(dataset, feature_to_find):
    """
    Get list of masks with given feature in the prompt
    """
    samples_list = dataset.get_samples_list()
    masks_list = []
    for sample in samples_list:
        prompt = sample['prompt']
        mask_filename = os.path.join(dataset.dataset_path, sample['source'])
        if feature_to_find in prompt:
            masks_list.append(mask_filename)

    print("Found {} masks with feature {}".format(len(masks_list), feature_to_find))

    return masks_list


diagnosed_masks = get_masks_list(dataset, feature_to_find="heart failure")
healthy_masks = get_masks_list(dataset, feature_to_find="healthy")


def generate_prompt(features):
    """
    Create random prompt from given features
    """

    prompt = ""
    for feature in features:
        chosen_value = random.choice(feature)
        prompt += ", " + chosen_value
    prompt = prompt[2:]

    return prompt


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps=20, guess_mode=False, strength=1, scale=9, seed=-1, eta=0):
    """
    Generate synthetic image from input mask and prompt
    """

    with torch.no_grad():
        
        img = input_image
        H, W, C = img.shape

        control = img.cuda()

        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
    return x_samples


def generate_synthetic_copy(dataset_path, output_path=None, per_sample_multiplier=1):
    """
    Generate synthetic version of dataset with real prompts and corresponding masks
    """

    out_dir = 'synthetic_dataset'

    if output_path is None:
        print(dataset_path)
        print(os.path.basename(dataset_path))
        output_name = os.path.basename(dataset_path) + "_synthetic"
        output_dir = os.path.join(out_dir, output_name)

    os.makedirs(output_dir, exist_ok=True)

    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=20, batch_size=1, shuffle=True)
    for item in tqdm(dataloader):
        jpg = item['jpg'][0]
        txt = item['txt'][0]
        hint = item['hint'][0]
        filename = item['filename'][0]
        samples = process(hint, txt, "", "", per_sample_multiplier, hint.shape[1])
        for i in range(samples.shape[0]):
            sample = samples[i]
            # sample = sample.transpose(1, 2, 0)
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            filename = Path(filename).stem
            output_path = os.path.join(output_dir, f'{filename}_synthetic_{i}.png')
            print(output_path)
            cv2.imwrite(output_path, sample)


def generate_random_prompt_dataset(output_dir, n_samples = 10):
    """
    Generate dataset with random prompts and real masks from patients diagnosed heart failure
    """

    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(n_samples)):
        prompt = generate_prompt(features)
        if "healthy" in prompt:
            mask = random.choice(healthy_masks)
        else:
            mask = random.choice(diagnosed_masks)

        mask = cv2.imread(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask.astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)

        sample = process(mask, prompt, "", "", 1, mask.shape[1])[0]

        filename = prompt.replace(", ", "_").replace(" ", "_")
        filename = f'{i}_{filename}.png'
        output_path = os.path.join(output_dir, filename)
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(mask.numpy() * 255.0, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, sample)
        cv2.imwrite(output_path.replace(".png", "_mask.png"), mask)
    
    print("Generated random prompt dataset with {} samples".format(n_samples))


# Select one of the methods for synthetic dataset generation

generate_random_prompt_dataset("synthetic_dataset/random_dataset", 10)
# generate_synthetic_copy(source_dataset_path)

