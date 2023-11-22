from share import *
import config
import os
from pathlib import Path

from tqdm import tqdm
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


source_dataset_path = "./training/stacked_EDES_resized_128"



# model_checkpoint = './lightning_logs/version_21/checkpoints/epoch=0-step=30296.ckpt' # less than 1 epoch
# model_checkpoint = 'lightning_logs/version_23/checkpoints/epoch=3-step=121187.ckpt' # 512
model_checkpoint = 'lightning_logs/version_43/checkpoints/epoch=18-step=71971.ckpt' # 128 

model = create_model('./models/cldm_v21.yaml').cuda()
model.load_state_dict(load_state_dict(model_checkpoint, location='cpu'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps=20, guess_mode=False, strength=1, scale=9, seed=-1, eta=0):
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
        samples = process(hint, txt, "", "", 1, hint.shape[1])
        for i in range(samples.shape[0]):
            sample = samples[i]
            # sample = sample.transpose(1, 2, 0)
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            filename = Path(filename).stem
            output_path = os.path.join(output_dir, f'{filename}_synthetic_{i}.png')
            print(output_path)
            cv2.imwrite(output_path, sample)


generate_synthetic_copy(source_dataset_path)