import torch
import numpy as np
model=torch.load("checkpoints_backup/checkpoint.5.pkl")
np.save('weights.npy',model._modules['conv1'].weight.detach().cpu().numpy())
