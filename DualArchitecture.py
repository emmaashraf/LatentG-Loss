import torch 
import torch.nn as nn
import torch.optim as optim

class DualTextCNN(nn.Module):
    def __init__(self, input_dim=300, num_classes=7, latent_dim=32):
        super(DualTextCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool1d(1)         
        )
        
        self.latent = nn.Linear(256, latent_dim)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes) 
        )

        self.decoder_ae = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        latent_x = self.latent(x)
        class_output = self.fc(latent_x)
        reconstructed_vector = self.decoder_ae(latent_x)

        return class_output, reconstructed_vector.unsqueeze(1), latent_x
