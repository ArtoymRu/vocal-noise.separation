import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import PyTorchModelHubMixin

# Import local modules
from .utils.one_path_flash_fsmn import Encoder, Decoder, Dual_Path_Model, SBFLASHBlock_DualA

def get_checkpoints(config_name):
    '''Downloads model checkpoints if they are not present locally.'''
    for file in ['encoder', 'decoder', 'masknet']:
        model_path = f'./models/{file}.pt'
        if not os.path.exists(model_path):
            print(f"Downloading {file}.pt")
            hf_hub_download(repo_id="username/mossformer2", filename=f"{file}.pt", cache_dir="./models")

# Define the model components and their interactions
class Mossformer2(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(Mossformer2, self).__init__()
        self.encoder = Encoder(config['encoder'])
        self.decoder = Decoder()
        self.dual_path_model = Dual_Path_Model()
        
    def forward(self, x):
        encoded = self.encoder(x)
        processed = self.dual_path_model(encoded)
        decoded = self.decoder(processed)
        return decoded
