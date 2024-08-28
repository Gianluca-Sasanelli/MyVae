#train
import os, datetime, torch, pickle,  argparse, yaml
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from model import VanillaVAE
from utils import CelebA, galaxy_Dataset


#Parser. Use --config $ConfigPath. Reading from yaml config file
parser = argparse.ArgumentParser()
parser.add_argument("--config", type = str, required = True, help = "config path")
args = parser.parse_args()
config_path = args.config
with open(config_path) as file:
    config = yaml.safe_load(file)
    
#data
device = "cuda"
dataset = config["dataset"]
if dataset == "CelebA":
    data_dir = r"C:\Users\Gianl\Desktop\celeba\img_align_celeba" #directory of images
elif dataset == "KiDS":
    data_dir = r"C:\Users\Gianl\Desktop\MyVae\data_dir\galaxies"

# Parameters
## Settings
settings_params = config["settings_parameters"]
image_size = settings_params["image_size"]
size_dataset = settings_params["size_dataset"]
precision = settings_params["precision"]
del settings_params
##logging parameters
logging_params = config["logging_parameters"]
out_dir = logging_params["out_dir"]
os.makedirs(out_dir, exist_ok=True)
log_interval = logging_params["log_interval"]
eval_iters = logging_params["eval_iters"]
init_from = logging_params["init_from"]
num_epochs = logging_params["num_epochs"]
best_val_loss = 1e9
del logging_params


##model parameters
model_params = config["model_parameters"]
in_channels = model_params["in_channels"]
latent_dimension = model_params["latent_dimension"]
hidden_dimensions = model_params["hidden_dimensions"]
del model_params
##Training parameters
training_params = config["training_parameters"]
batch_size = training_params["batch_size"]
kld_weight = training_params["kld_weight"]
max_lr = training_params["max_lr"]
gamma = training_params["gamma"]
# gradient_accomulation_iter = 1
weight_decay = training_params["weight_decay"]
decay_lr = training_params["decay_lr"]
grad_clip = training_params["grad_clip"]
del training_params
del config
#Defining the module
model_args = dict(in_channels = in_channels, latent_dim = latent_dimension, kld_weight = kld_weight, hidden_dims = hidden_dimensions)
#optimizer and scheduler
if init_from == "scratch":
    print("Initializing a new model from scratch")
    model = VanillaVAE(**model_args)
    resume_epoch = 0
elif init_from == "resume":
    print(f"Resuming from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location = device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in ['in_channels', 'latent_dim', 'kld_weight', 'hidden_dims']:
        model_args[k] = checkpoint_model_args[k]
    model = VanillaVAE(**model_args)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    resume_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]
    
model.to(device)
optimizer = model.configure_optimizer(weight_decay, learning_rate = max_lr)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None #flush the memory

#useful functions
transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

def get_batch(split):
    if split == "train":
        batch = next(iter(train_loader))
    if split == "val":
        batch = next(iter(val_loader))
    return batch.to(device)

def get_lr(iter):
        return max_lr * (gamma ** iter)
    
#Initialization of train and val set
# ------------------------------------
if dataset == "CelebA":
    data = CelebA(data_dir, transform = transform, size = size_dataset) # with size you can specify if you want the whole dataset or part of it
elif dataset == "KiDS":
    data = galaxy_Dataset(data_dir, transform = transform, size = size_dataset)
    
train_split = int(len(data) * 0.9)
val_split = len(data) - train_split
train_set, val_set = torch.utils.data.random_split(data, [train_split, val_split])
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, pin_memory = True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, pin_memory = True)
print("Length train loader:", len(train_loader))
# -------------------------------------
#scaler and ctx using bfloat16 data 
if precision == "bfloat16":
    ctx = torch.amp.autocast(device_type = device, dtype = torch.bfloat16)
    scaler = torch.cuda.amp.GradScaler(enabled = True)


@torch.no_grad()
#Valdiation pass
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        running_losses = torch.zeros((3,eval_iters))
        for k in range(eval_iters):
            data = get_batch(split)
            if precision == "bfloat16":
                with ctx:
                    _, loss = model(data)
            else:
                _, loss = model(data)
            running_losses[0,k] = loss["loss"].item()
            running_losses[1,k] = loss["Reconstruction loss"].item()
            running_losses[2,k] = loss["KLD loss"].item()
        out[split] = running_losses.mean(dim = 1)
    model.train()
    return out
#Training pss
def one_epoch_pass():
    model.train()
    last_loss = torch.zeros((3,))
    t0 = datetime.datetime.now()
    for i, batch in enumerate(train_loader):
        data = batch.to(device)
        if precision == "bfloat16":
            with ctx:
                _, loss_dict = model(data)
            loss  = loss_dict["loss"] 
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            _, loss_dict = model(data)
            loss  = loss_dict["loss"] 
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
        if i % log_interval == 0 and i != 0:
            last_loss[0], last_loss[1], last_loss[2] = loss.item(), loss_dict["Reconstruction loss"].item(), loss_dict["KLD loss"].item()
            t1 = datetime.datetime.now()
            dt = t1 -t0 
            print(f"step {i // log_interval}| loss {last_loss[0]:.5f}| reconstruction {last_loss[1]:.5f}| kld loss {last_loss[2]:.5f}| norm: {norm:.2f}| time {dt.seconds:.2f} s")
            t0 = t1
    return [last_loss[0],last_loss[1], last_loss[2]]

#TRAINING LOOP
train_losses = [[]]
val_losses = [[]]
print("Starting of the training loop")
for epoch in range(resume_epoch, num_epochs):
    lr = get_lr(epoch) if decay_lr else max_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("-----------------------------------------")
    print("lr:", lr)
    t0 = datetime.datetime.now()
    losses = estimate_loss()
    val_losses.append([losses["val"][0], losses["val"][1], losses["val"][2]])
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

    train_losses.extend(one_epoch_pass())
    t1 = datetime.datetime.now()
    dt = (t1 -t0)
    dt = dt.seconds / 60
    print(f"Epoch {epoch} ends, time of validation and training of one epochs: {dt:.1f} minutes")
    print("--------------------------------")
    if epoch > num_epochs - 1:
        with open(out_dir + os.sep + "train_losses.pkl", "wb") as file:
            pickle.dump(train_losses, file)
        with open(out_dir + os.sep +  "val_losses.pkl", "wb") as file:
            pickle.dump(val_losses, file)
        break
    
    
