import argparse
import safetensors.torch
from stable_diffusion_pytorch import convert_from_sdmodel

parser = argparse.ArgumentParser()
parser.add_argument(
    "--sd_model", type=str, required=True,
    help="Stable Diffusion model to load in safetensors format"
)
parser.add_argument(
    "--save_to", type=str, required=True,
    help="The prefix of the path to save to, for example \"./mymodel_\" " \
        "will save 4 files, mymodel_clip.safetensors, mymodel_decoder.safetensors, " \
        "mymodel_encoder.safetensors, and mymodel_diffusion.safetensors"
)

args = parser.parse_args()

# load the checkpoint file
state_dict = safetensors.torch.load_file(args.sd_model)

# convert to the state_dicts format that our library expects
state_dicts = convert_from_sdmodel.split_state_dict(state_dict)

for key in state_dicts:
    outPath = args.save_to + key + '.safetensors'
    print('Writing', outPath)
    safetensors.torch.save_file(state_dicts[key], outPath)
