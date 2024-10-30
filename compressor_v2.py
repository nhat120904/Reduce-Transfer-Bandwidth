import torch.nn as nn
from .network_components import ResnetBlock, FlexiblePrior, Downsample, Upsample
from .utils import quantize, NormalDistribution
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from compressai.datasets import Vimeo90kDataset
import torch.optim as optim
import torch
import torch.nn.functional as F
import math
import struct
import os
from pathlib import Path
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from compressai.datasets import ImageFolder
from torchvision import transforms
import tensorflow as tf
import tensorflow_compression as tfc
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import logging

def get_entropy_model_variables(entropy_model):
    """Returns the variables of the entropy model as numpy arrays."""
    return [var.numpy() for var in entropy_model.trainable_variables]

def compare_variables(prev_vars, current_vars):
    """Compares the previous and current variables, returns True if they are different."""
    for prev, curr in zip(prev_vars, current_vars):
        if not np.array_equal(prev, curr):
            return True
    return False

# Configure logging to log to a file
logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

def save_bottleneck_as_tfci(compressed_bitstream, output_file):
    print("check ypte 2: ", type(compressed_bitstream))
    if not output_file.endswith('.tfci'):
        output_file += ".tfci"
    # print("logging: ", b''.join(compressed_bitstream.numpy().tolist()))
    compressed_bitstream = b''.join(compressed_bitstream.numpy().tolist())
    if compressed_bitstream is None or not compressed_bitstream:
        print("Compression failed or resulted in an empty bitstream.")
        return
    try:
    # Save the compressed bitstream to a .tfci file
        with tf.io.gfile.GFile(output_file, "wb") as f:
            f.write(compressed_bitstream)
    except Exception as e:
        print("error", type(compressed_bitstream.numpy()))
        logging.error("An error occurred: %s", e)

    print(f"Bottleneck tensor compressed and saved to {output_file}")

# Convert PyTorch tensor to TensorFlow tensor (with dimension reordering)
def pytorch_to_tensorflow(tensor):
    # Permute from [batch_size, channels, height, width] (PyTorch) to [batch_size, height, width, channels] (TensorFlow)
    tensor = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
    return tf.convert_to_tensor(tensor)

# Convert TensorFlow tensor to PyTorch tensor (with dimension reordering)
def tensorflow_to_pytorch(tensor):
    # Convert back from [batch_size, height, width, channels] (TensorFlow) to [batch_size, channels, height, width] (PyTorch)
    tensor = tensor.numpy()
    return torch.tensor(tensor).permute(0, 3, 1, 2).float().to(device)

# entropy model
class EntropyModel(tf.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.prior = tfc.NoisyDeepFactorized(batch_shape=(num_filters,))
        self.entropy_model = tfc.ContinuousBatchedEntropyModel(
            prior=self.prior,
            coding_rank=3,
            compression=False  # Set to True for deployment
        )

    def compress(self, y):
        # Perform entropy modeling on the latent tensor `y`
        return self.entropy_model(y, training=True)

# Compressor model use with entropy model: Nhat
class Compressor(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))
        self.reversed_dims = [*map(lambda m: dim * m, reverse_dim_mults), out_channels]
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))
        assert self.dims[-1] == self.reversed_dims[0]
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)]
        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:]))
        self.reversed_hyper_dims = list(
            reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)])
        )
        self.reversed_hyper_in_out = list(
            zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:])
        )
        self.prior = FlexiblePrior(self.hyper_dims[-1])

    def get_extra_loss(self):
        return self.prior.get_extraloss()

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

    def extract_latent(self, input):
        for i, (resnet, down) in enumerate(self.enc):
            input = resnet(input)
            input = down(input)
        latent = input
        return latent
    
    def encode(self, input):
        for i, (resnet, down) in enumerate(self.enc):
            input = resnet(input)
            input = down(input)
        latent = input
        for i, (conv, act) in enumerate(self.hyper_enc):
            input = conv(input)
            input = act(input)
        hyper_latent = input
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        input = q_hyper_latent
        for i, (deconv, act) in enumerate(self.hyper_dec):
            input = deconv(input)
            input = act(input)

        mean, scale = input.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
        }
        return q_latent, q_hyper_latent, state4bpp

    def decode(self, input):
        output = []
        for i, (resnet, up) in enumerate(self.dec):
            input = resnet(input)
            input = up(input)
            output.append(input)
        return output[::-1]

    def extract_q_latent(self, q_hyper_latent, latent):
        input = q_hyper_latent

        for deconv, act in self.hyper_dec:
            input = deconv(input)
            input = act(input)

        mean, scale = input.chunk(2, 1)

        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))

        q_latent = quantize(latent, "dequantize", latent_distribution.mean)

        return q_latent

    def bpp(self, shape, state4bpp):
        B, _, H, W = shape
        latent = state4bpp["latent"]
        hyper_latent = state4bpp["hyper_latent"]
        latent_distribution = state4bpp["latent_distribution"]
        if self.training:
            q_hyper_latent = quantize(hyper_latent, "noise")
            q_latent = quantize(latent, "noise")
        else:
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        hyper_rate = -self.prior.likelihood(q_hyper_latent).log2()
        cond_rate = -latent_distribution.likelihood(q_latent).log2()
        bpp = (hyper_rate.sum(dim=(1, 2, 3)) + cond_rate.sum(dim=(1, 2, 3))) / (H * W)
        return bpp

    def forward(self, input):
        q_latent, q_hyper_latent, state4bpp = self.encode(input)
        bpp = self.bpp(input.shape, state4bpp)
        output = self.decode(q_latent)
        return {
            "output": output,
            "bpp": bpp,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
        }

class ResnetCompressor(Compressor):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
    ):
        super().__init__(
            dim,
            dim_mults,
            reverse_dim_mults,
            hyper_dims_mults,
            channels,
            out_channels
        )
        self.build_network()

    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        Downsample(dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                        Upsample(dim_out if not is_last else dim_in, dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )
            
def train(device):

    writer = SummaryWriter(log_dir='runs/ResnetCompressor')

    net = ResnetCompressor(
        dim=64,
        dim_mults=[1,2,3,4],
        reverse_dim_mults=[4,3,2,1],
        hyper_dims_mults=[4,4,4],
        channels=3,
        out_channels=64,
    )
    
    state_dict = torch.load('resnet_compressor_weights.pt')
    
    net.load_state_dict(state_dict, strict=False)
    net.to(device)
    net.eval()
    
    # freeze the encoder and decoder, just train the entropy model
    for param in net.enc.parameters():
        param.requires_grad = False
    for param in net.dec.parameters():
        param.requires_grad = False
    for param in net.hyper_enc.parameters():
        param.requires_grad = False
    for param in net.hyper_dec.parameters():
        param.requires_grad = False
            
    num_epochs = 30
    parameters = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
    optimizer = optim.Adam(parameters, lr=1e-4)
    lmbda = 1e-2
    
    #load dataset
    train_transforms = transforms.Compose(
        [transforms.RandomCrop((256, 256)), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop((256, 256)), transforms.ToTensor()]
    )

    root = "../../dataset/vimeo_septuplet"
    print(os.listdir(root))
    train_dataset = Vimeo90kDataset(root, transform=train_transforms, split='train', tuplet=7)    
    entropy_model = EntropyModel(num_filters = 256)
    #train model
    tf_optimizer = tf.optimizers.Adam(learning_rate=1e-3)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, drop_last = True)
    
    # Initial state of the entropy model variables
    previous_variables = get_entropy_model_variables(entropy_model.entropy_model) 
    
    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0
        
        for batch_idx, inputs in enumerate(train_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            out = net(inputs)
 
            # bitrate of the quantized latent
            N, _, H, W = inputs.size()
            num_pixels = N * H * W
            real_latent = out["q_latent"] #latent of input

            latent_tf = pytorch_to_tensorflow(real_latent)
            with tf.GradientTape() as tape:
                # Pass latent tensor through entropy model
                latent_tf_hat, bits = entropy_model.compress(latent_tf)

            # Compute gradients for the entropy model (TensorFlow)
            tf_gradients = tape.gradient(bits, entropy_model.entropy_model.trainable_variables)

            # Apply the gradients to the entropy model (TensorFlow)
            tf_optimizer.apply_gradients(zip(tf_gradients, entropy_model.entropy_model.trainable_variables))

            latent_hat = tensorflow_to_pytorch(latent_tf_hat)
            # reconstructed = compression_model.decode(latent_hat)
            bpp_loss = bits.numpy().sum() / num_pixels
            # print("bpp loss: ", bpp_loss)
            # mse_loss = F.mse_loss(real_latent, pred_latent)
            # print("mse loss: ", mse_loss)

            # final loss
            loss = bpp_loss
            # loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # aux_loss.backward()
            # aux_optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)
            writer.add_scalar('BPP/train', bpp_loss.item(), epoch * len(train_loader) + batch_idx)
            # writer.add_scalar('MSE/train', mse_loss.item(), epoch * len(train_loader) + batch_idx)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")

        current_variables = get_entropy_model_variables(entropy_model.entropy_model)
        if compare_variables(previous_variables, current_variables):
            print(f"Variables have changed after epoch {epoch + 1}")
        else:
            print(f"Variables have NOT changed after epoch {epoch + 1}")
        previous_variables = current_variables

        if (epoch + 1) % 1 == 0:
            checkpoint_path = f"compress_net_v2_checkpoint_epoch_{epoch+1}.pth"
            torch.save(net.state_dict(), checkpoint_path)
            
            # Save the entropy model's weights after training
            checkpoint = tf.train.Checkpoint(entropy_model=entropy_model.entropy_model)
            # Save checkpoint
            checkpoint_directory = f'./checkpoints/entropy_model_epoch_{epoch+1}'
            checkpoint.save(file_prefix=checkpoint_directory)
            print(f"Checkpoint saved: {checkpoint_path}")

    writer.close()
        
def get_header(model_name, metric, quality):
    """Format header information:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    # metric = metric_ids[metric]
    # code = (metric << 4) | (quality - 1 & 0x0F)
    return 0, 0  # model_ids[model_name], code

def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    
def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    
def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    
def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size
    
        
def inference(device):
    net = ResnetCompressor(
        dim=64,
        dim_mults=[1,2,3,4],
        reverse_dim_mults=[4,3,2,1],
        hyper_dims_mults=[4,4,4],
        channels=3,
        out_channels=64,
    )
    
    state_dict = torch.load("compress_net_v2_checkpoint_epoch_6.pth")
    net.load_state_dict(state_dict, strict=True)
    net.to(device)
    net.eval()
    
    img = torchvision.io.read_image("real.jpg")
    # img = torchvision.transforms.Resize((256, 256), antialias = None)(img)
    x = img.unsqueeze(0).float().to(device) / 255.0
    x = x * 2.0 - 1.0
    _, _, h, w = x.shape
    
    metric='mse'
    quality=1
    checkpoint_directory = './checkpoints/entropy_model_epoch_6' 
    checkpoint = tf.train.Checkpoint(entropy_model=net.entropy_bottleneck.entropy_model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
    output_file = 'latent_string.txt'
    # output = net(x)
    # latent = output["q_latent"]
    diff_latent = torch.load("diff.pt")
    diff_latent.to(device)
    diff_latent = pytorch_to_tensorflow(diff_latent)

    #compress bottleneck tensor
    y_strings = net.entropy_bottleneck.entropy_model.compress(diff_latent)
    print("check type: ", type(y_strings))

    out = {"strings": [y_strings.numpy()], "shape": tf.shape(diff_latent)[-2:]}
    shape = out["shape"]
    save_bottleneck_as_tfci(y_strings, "output")

    #decompress bit string
    y_hat = net.entropy_bottleneck.entropy_model.decompress(y_strings, [h, w])
    y_hat = tensorflow_to_pytorch(y_hat)
    # output = net.decode(y_hat)
    torch.save(y_hat, "tf_compress.pt")

    header = get_header(net, metric, quality)
    with Path(output_file).open("wb") as f:
        write_uchars(f, header)
        # write original image size
        write_uints(f, (h, w))
        # write shape and number of encoded latents
        write_uints(f, (shape[0], shape[1],  len(out["strings"])))
        for s in out["strings"]:
            write_uints(f, (len(s[0]),))
            write_bytes(f, s[0])
            
    size = filesize(output_file)
    bpp = float(size) * 8 / (h * w)
    print(f"{bpp:.4f} bpp ")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train(device)
    # inference(device)