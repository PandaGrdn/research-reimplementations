import torch
import torch.nn as nn

class SparseOvercompleteAutoencoder(nn.Module):
    def __init__(self, d_model, num_features):
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features
        self.W_enc = nn.Linear(d_model, num_features)
        self.W_dec = nn.Linear(num_features, d_model)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def encode(self, x):
        return torch.relu(self.W_enc(x - self.W_dec.bias))

    def decode(self, z):
        return self.W_dec(z)

    def normalize_decoder_(self):
        weight = self.W_dec.weight.data
        weight /= weight.norm(p=2, dim=0, keepdim=True).clamp(min=1e-8)

    def loss(self, x, x_hat, z, l1_coeff=0.01):
        recon = (x_hat - x).pow(2).mean()
        sparsity = z.abs().sum(dim=1).mean()
        return recon + l1_coeff * sparsity