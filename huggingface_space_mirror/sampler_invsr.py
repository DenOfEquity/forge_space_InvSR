#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-07-13 16:59:27

import os, sys, math, random

import importlib

import cv2
import numpy as np
from pathlib import Path

from utils import util_image
from utils import util_color_fix

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

import spaces

from pipeline_invsr import StableDiffusionInvEnhancePipeline, retrieve_timesteps

import noisepredictor


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def reload_model(model, ckpt):
    module_flag = list(ckpt.keys())[0].startswith('module.')
    compile_flag = '_orig_mod' in list(ckpt.keys())[0]

    for source_key, source_value in model.state_dict().items():
        target_key = source_key
        if compile_flag and (not '_orig_mod.' in source_key):
            target_key = '_orig_mod.' + target_key
        if module_flag and (not source_key.startswith('module')):
            target_key = 'module.' + target_key

        assert target_key in ckpt
        source_value.copy_(ckpt[target_key])


def get_torch_dtype(torch_dtype: str):
    if torch_dtype == 'torch.float16':
        return torch.float16
    elif torch_dtype == 'torch.bfloat16':
        return torch.bfloat16
    elif torch_dtype == 'torch.float32':
        return torch.float32
    else:
        raise ValueError(f'Unexpected torch dtype:{torch_dtype}')

class BaseSampler:
    def __init__(self, configs):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
                configs.sampler_config.{start_timesteps, padding_mod, seed, sf, num_sample_steps}
            seed: int, random seed
        '''
        self.configs = configs

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.configs.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def write_log(self, log_str):
        print(log_str, flush=True)

    def build_model(self):
        # Build Stable diffusion
        params = dict(self.configs.sd_pipe.params)
        torch_dtype = params.pop('torch_dtype')
        params['torch_dtype'] = get_torch_dtype(torch_dtype)
        base_pipe = get_obj_from_str(self.configs.sd_pipe.target).from_pretrained(**params)
        if self.configs.get('scheduler', None) is not None:
            pipe_id = self.configs.scheduler.target.split('.')[-1]
            self.write_log(f'Loading scheduler of {pipe_id}...')
            base_pipe.scheduler = get_obj_from_str(self.configs.scheduler.target).from_config(
                base_pipe.scheduler.config
            )
            self.write_log('Loaded Done')
        if self.configs.get('vae_fp16', None) is not None:
            params_vae = dict(self.configs.vae_fp16.params)
            torch_dtype = params_vae.pop('torch_dtype')
            params_vae['torch_dtype'] = get_torch_dtype(torch_dtype)
            pipe_id = self.configs.vae_fp16.params.pretrained_model_name_or_path
            self.write_log(f'Loading improved vae from {pipe_id}...')
            base_pipe.vae = get_obj_from_str(self.configs.vae_fp16.target).from_pretrained(
                **params_vae,
            )
            self.write_log('Loaded Done')
        if self.configs.base_model in ['sd-turbo', 'sd2base'] :
            sd_pipe = StableDiffusionInvEnhancePipeline.from_pipe(base_pipe)
        else:
            raise ValueError(f"Unsupported base model: {self.configs.base_model}!")
#        sd_pipe.to("cuda")
#        sd_pipe.enable_model_cpu_offload()
        spaces.automatically_move_pipeline_components(sd_pipe)
        if self.configs.sliced_vae:
            sd_pipe.vae.enable_slicing()
        if self.configs.tiled_vae:
            sd_pipe.vae.enable_tiling()
            sd_pipe.vae.tile_latent_min_size = self.configs.latent_tiled_size
            sd_pipe.vae.tile_sample_min_size = self.configs.sample_tiled_size
        # if self.configs.gradient_checkpointing_vae:
            # self.write_log(f"Activating gradient checkpointing for vae...")
            # sd_pipe.vae._set_gradient_checkpointing(sd_pipe.vae.encoder, True)
            # sd_pipe.vae._set_gradient_checkpointing(sd_pipe.vae.decoder, True)

        model_configs = self.configs.model_start
        params = model_configs.get('params', dict)
        model_start = noisepredictor.NoisePredictor(**params)
        model_start.cuda()
        ckpt_path = model_configs.get('ckpt_path')
        assert ckpt_path is not None
        self.write_log(f"Loading started model from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location=f"cuda")
        if 'state_dict' in state:
            state = state['state_dict']
        reload_model(model_start, state)
        self.write_log(f"Loading Done")
        model_start.eval()
        setattr(sd_pipe, 'start_noise_predictor', model_start)

        self.sd_pipe = sd_pipe

class InvSamplerSR(BaseSampler):
    @torch.no_grad()
    def sample_func(self, im_cond, _positive, _negative):
        '''
        Input:
            im_cond: b x c x h x w, torch tensor, [0,1], RGB
        Output:
            xt: h x w x c, numpy array, [0,1], RGB
        '''
        if self.configs.cfg_scale > 1.0:
            negative_prompt = [_negative,]
        else:
            negative_prompt = None

        prompt_embeds, negative_prompt_embeds = self.sd_pipe.encode_prompt(
            [_positive, ],
            'cuda',
            1,
            True if self.configs.cfg_scale > 1.0 else False,
            negative_prompt=negative_prompt,
        )

        ori_h_lq, ori_w_lq = im_cond.shape[-2:]
        ori_w_hq = ori_w_lq * self.configs.basesr.sf
        ori_h_hq = ori_h_lq * self.configs.basesr.sf
        vae_sf = (2 ** (len(self.sd_pipe.vae.config.block_out_channels) - 1))
        if hasattr(self.sd_pipe, 'unet'):
            diffusion_sf = (2 ** (len(self.sd_pipe.unet.config.block_out_channels) - 1))
        else:
            diffusion_sf = self.sd_pipe.transformer.patch_size
        mod_lq = vae_sf // self.configs.basesr.sf * diffusion_sf
        idle_pch_size = self.configs.basesr.chopping.pch_size

        total_pad_h_up = total_pad_w_left = 0
        if min(im_cond.shape[-2:]) < idle_pch_size:
            while min(im_cond.shape[-2:]) < idle_pch_size:
                pad_h_up = max(min((idle_pch_size - im_cond.shape[-2]) // 2, im_cond.shape[-2]-1), 0)
                pad_h_down = max(min(idle_pch_size - im_cond.shape[-2] - pad_h_up, im_cond.shape[-2]-1), 0)
                pad_w_left = max(min((idle_pch_size - im_cond.shape[-1]) // 2, im_cond.shape[-1]-1), 0)
                pad_w_right = max(min(idle_pch_size - im_cond.shape[-1] - pad_w_left, im_cond.shape[-1]-1), 0)
                im_cond = F.pad(im_cond, pad=(pad_w_left, pad_w_right, pad_h_up, pad_h_down), mode='reflect')
                total_pad_h_up += pad_h_up
                total_pad_w_left += pad_w_left

        if im_cond.shape[-2] == idle_pch_size and im_cond.shape[-1] == idle_pch_size:
            target_size = (
                im_cond.shape[-2] * self.configs.basesr.sf,
                im_cond.shape[-1] * self.configs.basesr.sf
            )
            res_sr = self.sd_pipe(
                image=im_cond.type(torch.float16),
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                target_size=target_size,
                timesteps=self.configs.timesteps,
                guidance_scale=self.configs.cfg_scale,
                output_type="pt",    # torch tensor, b x c x h x w, [0, 1]
            ).images
        else:
            if not (im_cond.shape[-2] % mod_lq == 0 and im_cond.shape[-1] % mod_lq == 0):
                target_h_lq = math.ceil(im_cond.shape[-2] / mod_lq) * mod_lq
                target_w_lq = math.ceil(im_cond.shape[-1] / mod_lq) * mod_lq
                pad_h = target_h_lq - im_cond.shape[-2]
                pad_w = target_w_lq - im_cond.shape[-1]
                im_cond= F.pad(im_cond, pad=(0, pad_w, 0, pad_h), mode='reflect')

            im_spliter = util_image.ImageSpliterTh(
                im_cond,
                pch_size=idle_pch_size,
                stride= int(idle_pch_size * 0.50),
                sf=self.configs.basesr.sf,
                weight_type=self.configs.basesr.chopping.weight_type,
                extra_bs=1 if self.configs.bs > 1 else self.configs.bs,
            )

            index = 0
            total = len(im_spliter)
            inputs=[]

            # set timesteps
            timesteps, num_inference_steps = retrieve_timesteps(self.sd_pipe.scheduler, self.configs.timesteps, 'cuda')
            latent_timestep = timesteps[:1].repeat(1)

            for image, index_infos in im_spliter:
                target_size = (
                    image.shape[-2] * self.configs.basesr.sf,
                    image.shape[-1] * self.configs.basesr.sf,
                )
                # preprocess image tile
                self.sd_pipe.image_processor.config.do_normalize = False
                image = self.sd_pipe.image_processor.preprocess(image)  # [0, 1], torch tensor, (b,c,h,w)
                self.sd_pipe.image_processor.config.do_normalize = True
                image_up = F.interpolate(image, size=target_size, mode='bicubic') # upsampling
                image_up = self.sd_pipe.image_processor.normalize(image_up)  # [-1, 1]

                # prepare latent variables
                if getattr(self.sd_pipe, 'start_noise_predictor', None) is not None:
                    with torch.amp.autocast('cuda'):
                        noise = self.sd_pipe.start_noise_predictor(
                            image, latent_timestep, sample_posterior=True, center_input_sample=True,
                        )
                else:
                    noise = None

                latents = self.sd_pipe.prepare_latents(
                    image_up, latent_timestep, 1, 1, prompt_embeds.dtype,
                    'cuda', noise, None,
                )
                index += 1
                print (f'InvSR: VAE encode: {index} of {total}', end='\r', flush=True)

                inputs.append(latents)
            print ('InvSR: VAE encode: done          ')

            outputs = []
            for image in inputs:
                result = self.sd_pipe(
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    latents=image,
                    timesteps=self.configs.timesteps,
                    guidance_scale=self.configs.cfg_scale,
                    output_type="latent",
                ).images

                outputs.append(result)

            del inputs

            #   reset spliter
            im_spliter = util_image.ImageSpliterTh(
                im_cond,
                pch_size=idle_pch_size,
                stride= int(idle_pch_size * 0.50),
                sf=self.configs.basesr.sf,
                weight_type=self.configs.basesr.chopping.weight_type,
                extra_bs=1 if self.configs.bs > 1 else self.configs.bs,
            )

            index = 0
            for im_lq_pch, index_infos in im_spliter:
                latent = outputs[index]

                image = self.sd_pipe.vae.decode(latent / self.sd_pipe.vae.config.scaling_factor, return_dict=False, generator=None)[0]
                image = self.sd_pipe.image_processor.postprocess(image, output_type="pt", do_denormalize=[True])    # torch tensor, b x c x h x w, [0, 1]

                im_spliter.update(image, index_infos)
                index += 1
                print (f'InvSR: VAE decode: {index} of {total}', end='\r', flush=True)

            del outputs
            print ('InvSR: VAE decode: done          ')

            res_sr = im_spliter.gather()

        total_pad_h_up *= self.configs.basesr.sf
        total_pad_w_left *= self.configs.basesr.sf
        res_sr = res_sr[:, :, total_pad_h_up:ori_h_hq+total_pad_h_up, total_pad_w_left:ori_w_hq+total_pad_w_left]

        if self.configs.color_fix:
            im_cond_up = F.interpolate(
                im_cond, size=res_sr.shape[-2:], mode='bicubic', align_corners=False, antialias=True
            )
            if self.configs.color_fix == 'ycbcr':
                res_sr = util_color_fix.ycbcr_color_replace(res_sr, im_cond_up)
            elif self.configs.color_fix == 'wavelet':
                res_sr = util_color_fix.wavelet_reconstruction(res_sr, im_cond_up)
            else:
                raise ValueError(f"Unsupported color fixing type: {self.configs.color_fix}")

        res_sr = res_sr.clamp(0.0, 1.0).cpu().permute(0,2,3,1).float().numpy()

        return res_sr

    def inference(self, image, positive, negative, bs=1):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''

        im_cond = image.astype(float) / 255.0  # h x w x c
        im_cond = util_image.img2tensor(im_cond).cuda()                   # 1 x c x h x w

        return self.sample_func(im_cond, positive, negative).squeeze(0)


if __name__ == '__main__':
    pass

