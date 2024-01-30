from share import *
from datetime import datetime
import os
import time
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader, WeightedRandomSampler
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Get gpu id from command line
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
gpus = [args.gpu]

# 24 1:05
# 8 1:40
# 1*2 6:05

model = '1.5'
# model = '2.1'
stage = 'SD'
# stage = 'control'
size = 512
balanced_sampling = True

if model == '2.1':
    model_definition = './models/cldm_v21.yaml'
    resume_path = './models/control_sd21_ini.ckpt'
    # resume_path = 'lightning_logs/version_21/checkpoints/epoch=0-step=30296.ckpt'
    # resume_path = 'lightning_logs/version_23/checkpoints/epoch=3-step=121187.ckpt'
    # resume_path = 'lightning_logs/version_43/checkpoints/epoch=18-step=71971.ckpt' # 128 

elif model == '1.5':
    model_definition = './models/cldm_v15.yaml'
    resume_path = './models/control_sd15_ini.ckpt'
    # resume_path = 'logs/Nov28_11-49-38_model_SD_1.5_128_lr_2e-06_sd_locked_False_control_locked_True/lightning_logs/version_0/checkpoints/epoch=1-step=7575.ckpt' # Full SD finetune
    # resume_path = 'logs/Nov28_15-16-30_model_SD_1.5_128_lr_2e-06_sd_locked_False_control_locked_True/lightning_logs/version_0/checkpoints/epoch=23-step=90911.ckpt' # Full SD moar epochs finetune
    # resume_path = 'logs/Nov29_13-05-17_model_SD_1.5_128_lr_1e-05_sd_locked_True_control_locked_False/lightning_logs/version_0/checkpoints/epoch=24-step=94699.ckpt' # Control finetune
    # resume_path = 'logs/Nov24_15-58-06_model_SD_1.5_128_lr_2e-06_sd_locked_False_control_locked_True/lightning_logs/version_0/checkpoints/epoch=17-step=51137.ckpt'

batch_size = 1
if size == 128:
    dataset_path = "./training/stacked_EDES_fold_0_prev_0_01_resized_128/"
elif size == 512:
    dataset_path = "./training/stacked_EDES_fold_0_prev_0_01_resized_512/"
else:
    raise Exception("Invalid size")

logger_freq = 300
only_mid_control = False

if stage == 'SD':
    sd_locked = False
    sd_locked_first_half = True
    control_locked = True
    learning_rate = 2e-6
elif stage == 'control':
    sd_locked = True
    control_locked = False
    learning_rate = 1e-5

appendix = "_model_SD_" + model + "_" + str(size) + "_lr_" + str(learning_rate) + "_sd_locked_" + str(sd_locked) + "sd_first_half_" + str(sd_locked_first_half) + "_control_locked_" + str(control_locked)
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
log_dir = os.path.join("logs", current_time + appendix)
os.makedirs(log_dir, exist_ok=True)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(model_definition).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.sd_locked_first_half = sd_locked_first_half
model.control_locked = control_locked
model.only_mid_control = only_mid_control

start_sleep_hour = 7
end_sleep_hour = 22

# Sleep callback
class TimeScheduleSleep(Callback):
    def __init__(self, start_sleep_hour, end_sleep_hour):
        self.start_sleep_hour = start_sleep_hour
        self.end_sleep_hour = end_sleep_hour

    def on_train_batch_start(self, *args, **kwargs):
        current_time = datetime.now().strftime("%H")
        current_time = int(current_time)
        if current_time >= start_sleep_hour and current_time < end_sleep_hour:
            print("Current time is " + str(current_time))
            # Sleep until end of office hours
            time_to_sleep = end_sleep_hour - current_time
            print("Going to sleep for " + str(time_to_sleep) + " hours")
            time.sleep(time_to_sleep * 60 * 60)


logger_params = dict(sample = True, plot_denoise_rows= False, plot_diffusion_rows= False)

# Misc
dataset = MyDataset(dataset_path)
if balanced_sampling:
    sampler = WeightedRandomSampler(dataset.sample_weights, len(dataset))
    dataloader = DataLoader(dataset, num_workers=20, batch_size=batch_size, sampler=sampler)
else:
    dataloader = DataLoader(dataset, num_workers=20, batch_size=batch_size, shuffle=True)

logger = ImageLogger(batch_frequency=logger_freq, log_images_kwargs=logger_params)
time_schedule_sleep = TimeScheduleSleep(start_sleep_hour, end_sleep_hour)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], default_root_dir=log_dir, accumulate_grad_batches=2)
# trainer = pl.Trainer(gpus=gpus, precision=32, callbacks=[logger, time_schedule_sleep], default_root_dir=log_dir, accumulate_grad_batches=2)


# Train!
trainer.fit(model, dataloader)
