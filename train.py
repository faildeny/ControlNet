from share import *
from datetime import datetime
import os


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# 24 1:05
# 8 1:40
# 1*2 6:05

model = '1.5'
# stage = 'SD'
stage = 'control'

if model == '2.1':
    model_definition = './models/cldm_v21.yaml'
    # resume_path = './models/control_sd21_ini.ckpt'
    # resume_path = 'lightning_logs/version_21/checkpoints/epoch=0-step=30296.ckpt'
    # resume_path = 'lightning_logs/version_23/checkpoints/epoch=3-step=121187.ckpt'
    # resume_path = 'lightning_logs/version_43/checkpoints/epoch=18-step=71971.ckpt' # 128 

elif model == '1.5':
    model_definition = './models/cldm_v15.yaml'
    # resume_path = './models/control_sd15_ini.ckpt'
    # resume_path = 'logs/Nov28_11-49-38_model_SD_1.5_128_lr_2e-06_sd_locked_False_control_locked_True/lightning_logs/version_0/checkpoints/epoch=1-step=7575.ckpt' # Full SD finetune
    resume_path = 'logs/Nov28_15-16-30_model_SD_1.5_128_lr_2e-06_sd_locked_False_control_locked_True/lightning_logs/version_0/checkpoints/epoch=23-step=90911.ckpt' # Full SD moar epochs finetune
    # resume_path = 'logs/Nov24_15-58-06_model_SD_1.5_128_lr_2e-06_sd_locked_False_control_locked_True/lightning_logs/version_0/checkpoints/epoch=17-step=51137.ckpt'

batch_size = 24
logger_freq = 300
only_mid_control = False

if stage == 'SD':
    sd_locked = False
    control_locked = True
    learning_rate = 2e-6
elif stage == 'control':
    sd_locked = True
    control_locked = False
    learning_rate = 1e-5

appendix = "_model_SD_" + model + "_128" + "_lr_" + str(learning_rate) + "_sd_locked_" + str(sd_locked) + "_control_locked_" + str(control_locked)
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
log_dir = os.path.join("logs", current_time + appendix)
os.makedirs(log_dir, exist_ok=True)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(model_definition).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.control_locked = control_locked
model.only_mid_control = only_mid_control

logger_params = dict(sample = True, plot_denoise_rows= False, plot_diffusion_rows= True)

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=20, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq, log_images_kwargs=logger_params)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], default_root_dir=log_dir, accumulate_grad_batches=1)


# Train!
trainer.fit(model, dataloader)
