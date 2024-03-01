# python3
# A demo of loading a third party model (everythingV5) meant for the original stable diffusion code, and generating images.
import os
import safetensors.torch
import torch
from stable_diffusion_pytorch import pipeline, checkpoint_loader

# Runs on intel gpu by default, to run on cuda remove the following import and change 'xpu' to 'cuda'.
import intel_extension_for_pytorch as ipex
device = 'xpu'


# where to store generated images
outDir = './out'

# path to SD 1.4/1.5 based model safetensors file
modelPath = '/stor/download2/anything_inkBase.safetensors'

# if true, use float16, otherwise float32
useHalfPrecision = True



os.makedirs(outDir, exist_ok=True)

# load the checkpoint file and convert to half precision if needed
state_dict = safetensors.torch.load_file(modelPath)
if useHalfPrecision:
	for x in state_dict:
		state_dict[x] = state_dict[x].half()

# convert to the state_dicts format that our library expects
state_dicts = checkpoint_loader.split_state_dict(state_dict)

# create the model objects, and apply the weights in state_dicts
models = checkpoint_loader.load_models(state_dicts, device, useHalfPrecision)

steps = 40
seed = 12345
prompt = '1girl,living room,silver leotard,navel,bunny girl,cute,silver hairs,black eyes,cleavage,(leaning_forward:1.6),cafe,black stocking,papilla,all fours'
negativePrompt = 'bad anatomy,bad hands,missing fingers,extra fingers,three hands,three legs,bad arms,missing legs,missing arms,poorly drawn face,bad face,fused face,cloned face,three crus,fused feet,fused thigh,extra crus,ugly fingers,horn,realistic photo,huge eyes,worst face,2girl,long fingers,disconnected limbs,worst quality,normal quality,low quality,low res,blurry,text,watermark,logo,banner,extra digits,cropped,jpeg artifacts,signature,username,error,sketch ,duplicate,ugly,monochrome,horror,geometry,mutation,disgusting'

fileName = prompt.replace(' ', '_').replace('\\', '＼').replace(':', '⦂').replace(',', '-') + '.' + str(seed) + '.png'

images = pipeline.generate([prompt], uncond_prompts=[negativePrompt],
			models=models, n_inference_steps=steps, seed=seed, device=device,
			height=768, width=512)

images[0].save(outDir + '/' + fileName)
