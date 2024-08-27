#Sample

import torch, os, pickle, torchvision
import matplotlib.pyplot as plt
from model import VanillaVAE

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

out_dir = "output"
device = "cuda"
num_grids= 3
num_images = 128
seed = 42
torch.manual_seed(seed)
ctx = torch.amp.autocast(device_type = "cuda", dtype = torch.bfloat16)
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location="cuda")
checkpoint_model_args = checkpoint["model_args"]
model = VanillaVAE(**checkpoint_model_args)
state_dict = checkpoint['model']
# model.load_state_dict(state_dict)
print("Model loaded")
print(**checkpoint_model_args)

# model.eval()
# model.to(device)
# with torch.no_grad():
#     with ctx:
#         fig, axes = plt.subplots(1, num_grids, figsize=(20 * num_grids, 10))
#         for i in range(num_grids):
#             samples = model.sample(num_samples = num_images, current_device = "cuda")
#             grid = torchvision.utils.make_grid(samples, nrow=20, normalize=True)
            
#             # Select the correct axis for the subplot
#             ax = axes[i] if num_grids > 1 else axes
            
#             ax.imshow(grid.permute(1, 2, 0).cpu().float().numpy())
#             ax.axis('off')  # Remove axes for a cleaner look

#         plt.show()