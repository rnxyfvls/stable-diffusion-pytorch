import torch
from . import Tokenizer, CLIP, Encoder, Decoder, Diffusion
from . import util
import warnings


def make_compatible(state_dict):
    keys = list(state_dict.keys())
    changed = False
    for key in keys:
        if "causal_attention_mask" in key:
            del state_dict[key]
            changed = True
        elif "_proj_weight" in key:
            new_key = key.replace('_proj_weight', '_proj.weight')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True
        elif "_proj_bias" in key:
            new_key = key.replace('_proj_bias', '_proj.bias')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
            changed = True

    if changed:
        warnings.warn(("Given checkpoint data were modified dynamically by make_compatible"
                       " function on model_loader.py. Maybe this happened because you're"
                       " running newer codes with older checkpoint files. This behavior"
                       " (modify old checkpoints and notify rather than throw an error)"
                       " will be removed soon, so please download latest checkpoints file."))

    return state_dict

# The following functions load the default models, used when the user supplied `models` is missing
# one or more of (clip, encoder, decoder, diffusion).

def load_clip(device, dtype):
    state_dict = torch.load(util.get_file_path('ckpt/clip.pt'))
    state_dict = make_compatible(state_dict)

    clip = CLIP().to(dtype).to(device)
    clip.load_state_dict(state_dict)
    return clip

def load_encoder(device, dtype):
    state_dict = torch.load(util.get_file_path('ckpt/encoder.pt'))
    state_dict = make_compatible(state_dict)

    encoder = Encoder().to(dtype).to(device)
    encoder.load_state_dict(state_dict)
    return encoder

def load_decoder(device, dtype):
    state_dict = torch.load(util.get_file_path('ckpt/decoder.pt'))
    state_dict = make_compatible(state_dict)

    decoder = Decoder().to(dtype).to(device)
    decoder.load_state_dict(state_dict)
    return decoder

def load_diffusion(device, dtype):
    state_dict = torch.load(util.get_file_path('ckpt/diffusion.pt'))
    state_dict = make_compatible(state_dict)

    diffusion = Diffusion().to(dtype).to(device)
    diffusion.load_state_dict(state_dict)
    return diffusion

r"""
    Create and load the 4 default models (clip, encoder, decoder, diffusion).
    Args:
        device (`str`):
            The device to run the models on, passed to model.to()
        dtype (`torch.dtype`, *optional*):
            The data type of the model to create. Note that this can be different
            from the data type of the checkpoint file, and pytorch will auto convert.
    Returns:
        `Dict[str, torch.nn.Module]`:
            The loaded models to be passed to pipeline.generate()
    """
def preload_models(device, dtype=torch.float32):
    return {
        'clip': load_clip(device, dtype),
        'encoder': load_encoder(device, dtype),
        'decoder': load_decoder(device, dtype),
        'diffusion': load_diffusion(device, dtype),
    }

r"""
    Create the 4 models that the pipeline expects and load the weights from state_dicts
    (not an original stable diffusion state_dict!).
    Args:
        state_dicts (`Dict[str, str]`):
            A dict with 4 keys: clip, encoder, decoder, diffusion; each key's value is a dict of weights
            for that model. You can pass in the dict returned by convert_from_sdmodel.split_state_dict().
        device (`str`):
            The device to run the models on, passed to model.to()
        dtype (`torch.dtype`, *optional*):
            The data type of the model to create. Note that this can be different
            from the data type of the checkpoint file, and pytorch will auto convert.
    Returns:
        `Dict[str, torch.nn.Module]`:
            The loaded models to be passed to pipeline.generate()
    """
def load_models(state_dicts, device, dtype=torch.float32):
    models = {}
    models['clip'] = CLIP().to(dtype).to(device)
    models['encoder'] = Encoder().to(dtype).to(device)
    models['decoder'] = Decoder().to(dtype).to(device)
    models['diffusion'] = Diffusion().to(dtype).to(device)

    models['clip'].load_state_dict(state_dicts['clip'])
    models['encoder'].load_state_dict(state_dicts['encoder'])
    models['decoder'].load_state_dict(state_dicts['decoder'])
    models['diffusion'].load_state_dict(state_dicts['diffusion'])
    return models
