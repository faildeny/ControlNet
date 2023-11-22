from share import *
from datetime import datetime
import os
import socket


import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# 24 1:05
# 8 1:40
# 1*2 6:05

# Configs
# resume_path = './models/control_sd21_ini.ckpt'
# resume_path = 'lightning_logs/version_21/checkpoints/epoch=0-step=30296.ckpt'
resume_path = 'lightning_logs/version_43/checkpoints/epoch=18-step=71971.ckpt' # 128 



batch_size = 24
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

appendix = "_512"
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
log_dir = os.path.join("logs", current_time + appendix)
os.makedirs(log_dir, exist_ok=True)

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

logger_params = dict(sample = True, plot_denoise_rows= False, plot_diffusion_rows= True)

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=20, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq, log_images_kwargs=logger_params)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], default_root_dir=log_dir, accumulate_grad_batches=1)


# Train!
trainer.fit(model, dataloader)
