from argparse import ArgumentParser

import torch
from PIL import Image
import glob
from tqdm import tqdm
import os
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor

# source data
image_folder = "E:\\Dokumente\\001_Studium\\Bachelor\\Programmieren\\grafics_CuttedImages_1400x1360\\"
image_appendix = '.tif'
image_output_folder_test = "E:\\Dokumente\\001_Studium\\Bachelor\\Programmieren\\test_data_144x128_GSD16"
image_output_folder_train = "E:\\Dokumente\\001_Studium\\Bachelor\\Programmieren\\train_data_144x128_GSD16"

# getting all images from one folder
image_list = []
for filename in glob.glob(image_folder + '*' + image_appendix):
    im = Image.open(filename)
    image_list.append(im)
print("read %d Images to image_list" % len(image_list))

sub_image_number_width = 144
sub_image_number_height = 128


# Algorithm
def cutting_image_n_times(image, image_nr, storage, tile_width, tile_height, image_apdx):
    width, height = image.size
    tiles = []
    for x in tqdm(range(0, width, tile_width)):
        for y in range(0, height, tile_height):
            if x + tile_width <= width and y + tile_height <= height:
                tile = image.crop((x, y, x + tile_width, y + tile_height))
                tiles.append(tile)
                tile_path = os.path.join(storage, f"image_Nr-{image_nr}_{x}_{y}{image_apdx}")
                tile.save(tile_path)
    return tiles


# Cutting all folder images
for i in tqdm(range(len(image_list))):
    cutting_image_n_times(image_list[i], i, image_output_folder_train, sub_image_number_width, sub_image_number_height,
                          image_appendix)


# -------------------------------------
class Parameters:
    def __init__(self):
        self.input_shape = (3, 128, 144)
        self.lr = 0.0003724005169734309  # https://towardsdatascience.com/how-to-choose-the-optimal-learning-rate-for-neural-networks-362111c5c783
        self.weight_decay = 1e-05
        self.batch_size = 200
        self.num_epochs = 10000
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None


P = Parameters()
torch.manual_seed(0)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ch, self.h, self.w = P.input_shape  # Features: 55296

        self.enc_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),  # Output: 16x64x72
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: 32x16x18
            nn.BatchNorm2d(32),
            nn.ReLU(),  # https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64x8x9
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(18432, 11060),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.enc_layers(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


# -------------------------------------
def main():
    encoder = Encoder()
    encoder.eval()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data", help="Data root directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Logging directory")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers > 0 turns on multi-process data loading")
    parser.add_argument("--image_size", type=int, default=128, help="Spatial size of training images")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size during training")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images")
    parser.add_argument("--nz", type=int, default=256, help="Size of latent vector z")
    parser.add_argument("--nfe", type=int, default=64, help="Size of feature maps in encoder")
    parser.add_argument("--nfd", type=int, default=64, help="Size of feature maps in decoder")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizer")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 hyperparameter for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizer")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs. Use 0 for CPU mode")

    args = parser.parse_args()
    main(args)
