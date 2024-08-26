import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaVAE(nn.Module):
    def __init__(self,
                in_channels: int,
                latent_dim: int,
                kld_weight: float,
                hidden_dims= None):
        super().__init__()

        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        

        modules = []
        #dimensionality of the features map at each step
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512, 1024]
        self.final_hidden_dim = hidden_dims[-1]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                    kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.SiLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # linear layer to obeain mean and variance of the distribution
        self.lin = nn.Linear(hidden_dims[-1]*4, latent_dim * 2)
        # starting the reconstruction of the output
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()
        # building the decoder
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.SiLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.SiLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                    kernel_size= 3, padding= 1),
                            nn.Tanh())
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.ConvTranspose2d):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                

    def encode(self, input: torch.Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the mean and variance of the latent gaussian distribution.
        :param input: (torch.Tensor) Input torch.Tensor to encoder [N x C x H x W]
        :return: (torch.Tensor) List of the parameters
        """
        latent = self.encoder(input)
        latent = torch.flatten(latent, start_dim=1)
        mu, log_var = torch.chunk(self.lin(latent), 2, dim = -1)
        #clamping to prevent the kl loss to diverge
        log_var = torch.clamp(log_var, min = -2, max = 2)
        return mu, log_var

    def decode(self, z: torch.Tensor):
        """
        Maps the given latent z into the image space.
        :param z: (torch.Tensor) [B x D]
        :return: (torch.Tensor) [B x C x H x W]
        """
        reconstruction= self.decoder_input(z)
        reconstruction = reconstruction.view(-1, self.final_hidden_dim, 2, 2)
        reconstruction = self.decoder(reconstruction)
        reconstruction = self.final_layer(reconstruction)
        return reconstruction

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, 
                loss = True):
        """
        Foward of the model. The original image is compressed into a latent value
        and then reconstructed. The computation of the loss can be activated throug the boolean loss.
        :input: (torch.Tensor) Starting Image [B x C x H x W]
        : loss: (Boolean) Whether the loss is computed
        
        """       
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        if loss:
            losses= self.loss_function(x_hat, input, mu, log_var)
        else:
            losses = None
        return x_hat, losses
    
    def configure_optimizer(self, weight_decay, learning_rate):
        # Getting parameters with requires_grad = True
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in convolution, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optimizer = torch.optim.Adam(optim_groups, lr=learning_rate)
        return optimizer    


    def loss_function(self,recons, target, mu, logvar):
        """
        Computes the VAE loss function
        :recons: (torch.Tensor) Reconstructed Image
        :target: (torch.Tensor) Starting image
        :mu: (torch.Tensor) Mean of the latent Gaussian 
        :var: (torch.Tensor) Variance of the latent Gaussian
        """
        
        recons_loss =F.mse_loss(recons, target)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        weighted_kld_loss = self.kld_weight * kld_loss

        loss = recons_loss + weighted_kld_loss
        
        return {"loss": loss, "Reconstruction loss": recons_loss.detach(), "KLD loss": weighted_kld_loss.detach()}

    def sample(self,
            num_samples:int,
            current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (torch.Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
        