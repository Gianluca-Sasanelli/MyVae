#train
import os, datetime, torch, pickle
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from model import VanillaVAE

device = "cuda"
data_dir = r"C:\Users\Gianl\Desktop\celeba\img_align_celeba" #directory of images
data_size = 64

# Parameters
##logging parameters
out_dir = "output"
log_interval = 20
eval_iters = 1
##model parameters
in_channels = 3
latent_dimension = 128
hidden_dimensions = [32, 64, 128, 256, 512]
##Train parameters
batch_size = 1024
kld_weight = 0.00025
max_lr = 0.005
gamma = 0.95
# gradient_accomulation_iter = 1
num_epochs = 100
weight_decay = 0.0
decay_lr = True
grad_clip = 1.0
#Defining the module
model_args = dict(in_channels = in_channels, latent_dim = latent_dimension, kld_weight = kld_weight, hidden_dims = hidden_dimensions)
print("Initializing a new model from scratch")
model = VanillaVAE(**model_args)
model.to(device)
#optimizer and scheduler
optimizer = model.configure_optimizer(weight_decay, learning_rate = max_lr)
print("Initializing optimizer")
#loaders
#useful functions
transform = transforms.Compose([
            transforms.Resize((data_size, data_size)),
            transforms.ToTensor(),
        ])

class CelebA(torch.utils.data.Dataset):

    def __init__(self, root, transform= None, size: float = 1.0):
        self.root = root
        self.transform = transform
        self.img_dir = os.path.join(self.root, 'img_align_celeba')
        self.fns = os.listdir(self.img_dir)
        self.size = size

    def __len__(self):
        return int(len(self.fns) * self.size)

    def __getitem__(self, index):
        fn = self.fns[index]
        img_path = os.path.join(self.img_dir, fn)
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img
     
def get_batch(split):
    if split == "train":
        batch = next(iter(train_loader))
    if split == "val":
        batch = next(iter(val_loader))
    return batch.to(device)

def get_lr(iter):
        return max_lr * (gamma ** iter)
    
#Initialization of train and val set
data = CelebA(data_dir, transform = transform, size = 1) # with size you can specify if you want the whole dataset or part of it
train_split = int(len(data) * 0.9)
val_split = len(data) - train_split
train_set, val_set = torch.utils.data.random_split(data, [train_split, val_split])
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, pin_memory = True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, pin_memory = True)
print("Length train loader:", len(train_loader))
print("Getting the data")
best_val_loss = 1e9
#scaler and ctx
ctx = torch.amp.autocast(device_type = device, dtype = torch.bfloat16)
scaler = torch.cuda.amp.GradScaler(enabled = True)
print("Scaler done")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        running_losses = torch.zeros((3,eval_iters))
        for k in range(eval_iters):
            data = get_batch(split)
            with ctx:
                _, loss = model(data)
            running_losses[0,k] = loss["loss"].item()
            running_losses[1,k] = loss["Reconstruction loss"].item()
            running_losses[2,k] = loss["KLD loss"].item()
        print(f"Split: {split}, running loss: {running_losses}")
        out[split] = running_losses.mean(dim = 1)
    model.train()
    return out

def one_epoch_pass():
    model.train()
    last_loss = torch.zeros((3,))
    t0 = datetime.datetime.now()
    for i, batch in enumerate(train_loader):
        data = batch.to(device)
        with ctx:
            _, loss_dict = model(data)
        loss  = loss_dict["loss"] 
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if i % log_interval == 0 and i != 0:
            last_loss[0], last_loss[1], last_loss[2] = loss.item(), loss_dict["Reconstruction loss"].item(), loss_dict["KLD loss"].item()
            t1 = datetime.datetime.now()
            dt = t1 -t0 
            print(f"step {i // log_interval}| loss {last_loss[0]:.5f}| reconstruction {last_loss[1]:.5f}| kld loss {last_loss[2]:.5f}| time {dt.seconds:.2f} s")
            t0 = t1
            last_loss.zero_()
    return last_loss[0]

#TRAINING LOOP
# data = get_batch("train")
# num_of_images_per_step = gradient_accomulation_iter * batch_size
train_losses = []
val_losses = []
print("Starting of the training loop")
for epoch in range(num_epochs):
    lr = get_lr(epoch) if decay_lr else max_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("-----------------------------------------")
    print("lr:", lr)
    losses = estimate_loss()
    val_losses.append(losses["val"][0])
    print(f"val | epoch {epoch}| train tot loss {losses['train'][0]:.4f}| train recons loss {losses['train'][1]:.4f}| train kld loss {losses['train'][2]:.2e}|  val tot loss {losses['val'][0]:.4f}| val recons loss {losses['val'][1]:.4f}| val kld loss {losses['val'][2]:.2e}")
    if losses["val"][0] < best_val_loss:
            best_val_loss = losses["val"][0]
            if epoch > 0:
                checkpoint = {
                    "model" : model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "model_args": model_args,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                }    
                print(f"save checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            with open(out_dir + os.sep + "train_losses.pkl", "wb") as file:
                    pickle.dump(train_losses, file)
            with open(out_dir + os.sep +  "val_losses.pkl", "wb") as file:
                    pickle.dump(val_losses, file)
    #gradient accumulation        

    one_epoch_pass()
    print(f"Epoch {epoch} ends")
    print("--------------------------------")
    if epoch > num_epochs - 1:
        with open(out_dir + os.sep + "train_losses.pkl", "wb") as file:
            pickle.dump(train_losses, file)
        with open(out_dir + os.sep +  "val_losses.pkl", "wb") as file:
            pickle.dump(val_losses, file)
        break
    
    
