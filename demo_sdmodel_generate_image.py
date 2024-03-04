# python3
# A demo of loading a third party model (everythingV5) meant for the original stable diffusion code, and generating images.
import os
import safetensors.torch
import torch
from stable_diffusion_pytorch import pipeline, convert_from_sdmodel, model_loader

# Runs on intel gpu by default, to run on cuda remove the following import and change 'xpu' to 'cuda'.
import intel_extension_for_pytorch as ipex
device = 'xpu'


# where to store generated images
outDir = './out'

# path to SD 1.4/1.5 based model safetensors file
modelPath = './sd-v1-5.safetensors' # download from https://huggingface.co/tabtap/sd-v1-5.safetensor/tree/main

# either float16 or float32
dtype = torch.float16



os.makedirs(outDir, exist_ok=True)

# load the checkpoint file
state_dict = safetensors.torch.load_file(modelPath)

# convert to the state_dicts format that our library expects
state_dicts = convert_from_sdmodel.split_state_dict(state_dict)

# create the model objects, and apply the weights in state_dicts
models = model_loader.load_models(state_dicts, device, dtype)

steps = 40
seed = 12345
prompt = '1girl,cirno,anime style'
negativePrompt = 'bad anatomy,bad hands,missing fingers,extra fingers'

fileName = prompt.replace(' ', '_').replace('\\', '＼').replace(':', '⦂').replace(',', '-') + '.' + str(seed) + '.png'

images = pipeline.generate([prompt], uncond_prompts=[negativePrompt],
			models=models, n_inference_steps=steps, seed=seed, device=device,
			height=768, width=512)

images[0].save(outDir + '/' + fileName)
