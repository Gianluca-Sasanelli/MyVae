#train
import os, math, time, torch, torchvision, pickle
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from model import VanillaVAE

device = "cuda"
data_dir = r"C:\Users\Gianl\Desktop\celeba\img_align_celeba" #directory of images


# Parameters
##logging parameters
out_dir = "output"
eval_interval = 200
log_interval = 100
eval_iters = 10
##model parameters
in_channels = 3
latent_dimension = 128

##Train parameters
batch_size = 256
kld_weight = batch_size / 200000
max_lr = 0.005
gamma = 0.95
gradient_accomulation_iter = 8
weight_decay = 0.0
decay_lr = True
max_iters = 5000
grad_clip = 1.0
#Defining the module
model_args = dict(in_channels = in_channels, latent_dim = latent_dimension, kld_weight = kld_weight)
print("Initializing a new model from scratch")
model = VanillaVAE(**model_args)
model.to(device)
#optimizer and scheduler
optimizer = model.configure_optimizer(weight_decay, learning_rate = max_lr)

#loaders
#useful functions
transform = transforms.Compose([
            transforms.Resize((128,128)),
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

def normalize(image, numpy = False):
    image = image.cpu()
    
    v_max = torch.max(image)
    v_min = torch.min(image)
    image = (image - v_min)/ (v_max - v_min)
    if numpy:
        image = image.numpy()
    return  image
    
    
def get_batch(split):
    if split == "train":
        batch = next(iter(train_loader))
    if split == "val":
        batch = next(iter(val_loader))
    
    return batch.to(device)

def get_lr(iter):
        return max_lr * (gamma ** iter)
    
data = CelebA(data_dir, transform = transform, size = 1) # with size you can specify if you want the whole dataset or part of it
train_split = int(len(data) * 0.9)
val_split = len(data) - train_split
train_set, val_set = torch.utils.data.random_split(data, [train_split, val_split])
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, pin_memory = True )
val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, pin_memory = True)

iter_num = 0
best_val_loss = 1e9
#scaler and ctx
ctx = torch.amp.autocast(device_type = device, dtype = torch.bfloat16)
scaler = torch.cuda.amp.GradScaler(enabled = True)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros((3,eval_iters))
        for k in range(eval_iters):
            data = get_batch(split)
            with ctx:
                _, loss = model(data)
            losses[0,k] = loss["loss"].item()
            losses[1,k] = loss["Reconstruction loss"].item()
            losses[2,k] = loss["KLD loss"].item()
        out[split] = losses.mean(dim = 1)
    model.train()
    return out
            
#TRAINING LOOP
data = get_batch("train")
t0 = time.time()
num_of_images_per_step = gradient_accomulation_iter * batch_size
train_losses = []
val_losses = []
while True:
    if iter_num % 100 == 0:
        lr = get_lr(iter_num) if decay_lr else max_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    if iter_num % eval_interval == 0 :
        losses = estimate_loss()
        val_losses.append(losses["val"][0])
        print(f"val | step {iter_num}| train tot loss {losses['train'][0]:.4f}| train recons loss {losses['train'][1]:.4f}| train kld loss {losses['train'][2]:.2e}|  val tot loss {losses['val'][0]:.4f}| val recons loss {losses['val'][1]:.4f}| val kld loss {losses['val'][2]:.2e}")
    if losses["val"][0] < best_val_loss:
            best_val_loss = losses["val"][0]
            if iter_num > 0:
                checkpoint = {
                    "model" : model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }    
                print(f"save checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            with open(out_dir + os.sep + "train_losses.pkl", "wb") as file:
                    pickle.dump(train_losses, file)
            with open(out_dir + os.sep +  "val_losses.pkl", "wb") as file:
                    pickle.dump(val_losses, file)
    #gradient accumulation        
    for micro_step in range(gradient_accomulation_iter):
        with ctx:
            _,  loss_dict = model(data)
            loss = loss_dict["loss"]
            loss = loss / gradient_accomulation_iter
            recons_loss = loss_dict["Reconstruction loss"]
            kld_loss = loss_dict["KLD loss"]
        data = get_batch("train")
        scaler.scale(loss).backward()
    #clip the gradient
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    t1 = time.time()
    dt = (t1-t0)
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accomulation_iter
        train_losses.append(lossf)
        completion_time = (max_iters - iter_num)*dt/3600
        print(f"step {iter_num}| total loss {lossf:.8f}| last recon loss {recons_loss:.7f}| last kld loss {kld_loss:.3f} | lr: {lr:.6f}| num_of_images_processed_per_step: {num_of_images_per_step}| Norm of the gradient: {norm:.1f}| Expected time left : {completion_time:.1f} hrs")    
    iter_num +=1
    if iter_num > max_iters:
        with open(out_dir + os.sep + "train_losses.pkl", "wb") as file:
            pickle.dump(train_losses, file)
        with open(out_dir + os.sep +  "val_losses.pkl", "wb") as file:
            pickle.dump(val_losses, file)
        break
    
    
