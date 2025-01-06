#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2024-12-11 17:17:41

from diffusers.utils import check_min_version
check_min_version("0.32.0")

import spaces
import warnings
warnings.filterwarnings("ignore")

import os
from urllib.parse import urlparse
from torch.hub import download_url_to_file

import numpy as np
import gradio as gr
from pathlib import Path
from omegaconf import OmegaConf
from sampler_invsr import InvSamplerSR

from utils import util_image

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Reference: https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def mkdir(dir_path, delete=False, parents=True):
    import shutil
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)
    if delete:
        if dir_path.exists():
            shutil.rmtree(str(dir_path))
    if not dir_path.exists():
        dir_path.mkdir(parents=parents)


def get_configs(num_steps=2, step_start=100, chopping_size=128, seed=12345):
    configs = OmegaConf.load("./configs/sample-sd-turbo.yaml")

    match num_steps:
        case 1:
            configs.timesteps = [step_start]
        case 2:
            configs.timesteps = [step_start, step_start//2]
        case 3:
            configs.timesteps = [step_start, step_start//2, step_start//4] #200, 100, 50
        case 4:
            configs.timesteps = [step_start, step_start//2, step_start//4, step_start//8] #200, 150, 100, 50
        case 5:
            configs.timesteps = [step_start, step_start//2, step_start//4, step_start//8, step_start//16] # 250, 200, 150, 100, 50
        case _:
            assert num_steps <= 250
            configs.timesteps = np.linspace(
                start=step_start, stop=0, num=num_steps, endpoint=False, dtype=np.int64()
            ).tolist()
            
    print(f'Setting timesteps for inference: {configs.timesteps}')

    # path to save noise predictor
    started_ckpt_name = "noise_predictor_sd_turbo_v5.pth"
    started_ckpt_dir = "./weights"
    mkdir(started_ckpt_dir, delete=False, parents=True)
    started_ckpt_path = Path(started_ckpt_dir) / started_ckpt_name
    if not started_ckpt_path.exists():
        load_file_from_url(
            url="https://huggingface.co/OAOA/InvSR/resolve/main/noise_predictor_sd_turbo_v5.pth",
            model_dir=started_ckpt_dir,
            progress=True,
            file_name=started_ckpt_name,
        )
    configs.model_start.ckpt_path = str(started_ckpt_path)

    configs.bs = 1
    configs.seed = seed 
    configs.basesr.chopping.pch_size = chopping_size
    if chopping_size == 128:
        configs.basesr.chopping.extra_bs = 8
    elif chopping_size == 256:
        configs.basesr.chopping.extra_bs = 4
    else:
        configs.basesr.chopping.extra_bs = 1

    return configs

def predict(image, num_steps=1, step_start=100, chopping_size=128, seed=12345):
    configs = get_configs(num_steps=num_steps, step_start=step_start, chopping_size=chopping_size, seed=seed)

    sampler = InvSamplerSR(configs)

    out_dir = Path('invsr_output')
    if not out_dir.exists():
        out_dir.mkdir()
    im_sr = sampler.inference(image, out_path=out_dir, bs=1)

    return im_sr

article = r"""
---

If you've found InvSR useful for your research or projects, please show your support by ⭐ the <a href='https://github.com/zsyOAOA/InvSR' target='_blank'>Github Repo</a>. Thanks!
---

If InvSR is useful for your research, please consider citing:
```bibtex
@article{yue2024InvSR,
  title={Arbitrary-steps Image Super-resolution via Diffusion Inversion},
  author={Yue, Zongsheng and Kang, Liao and Loy, Chen Change},
  journal = {arXiv preprint arXiv:2412.09013},
  year={2024},
}
```

📋 **License**

This project is licensed under <a rel="license" href="https://github.com/zsyOAOA/InvSR/blob/master/LICENSE">S-Lab License 1.0</a>.
Redistribution and use for non-commercial purposes should follow this license.

based on HuggingFace Space: https://huggingface.co/spaces/OAOA/InvSR
"""

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Arbitrary-steps Image Super-resolution via Diffusion Inversion.
    """)
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="numpy", label="Input: Low Quality Image", height="40vh")
            with gr.Row():
                steps = gr.Slider(
                    minimum=1, maximum=10, step=1,
                    value=1,
                    label="Number of steps",
                    )
                start = gr.Slider(
                    minimum=16, maximum=400, step=1,
                    value=100,
                    label="Start timestep",
                )
            chop = gr.Dropdown(
                choices=[128, 256, 512],
                value=128,
                label="Chopping size (for larger images: 1k->4k, try 256)",
                )
            seed = gr.Number(value=12345, precision=0, label="Seed")
            
        with gr.Column():
            generate = gr.Button(value="Generate")
            result = gr.Image(type="numpy", label="Output: High Quality Image", height="70vh", interactive=False)

    gr.Markdown(article)

    generate.click(fn=predict, inputs=[image, steps, start, chop, seed], outputs=[result])

if __name__ == "__main__":
    demo.launch()
