import torch
import torch.nn as nn
import torch.nn.init as init

def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)      


class TextAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(TextAutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim) # Latent vector 
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

        self.apply(weights_init)


    def forward(self, x):
        z = self.encoder(x) # z is latent vector.
        x_recon = self.decoder(z)
        return x_recon, z
    