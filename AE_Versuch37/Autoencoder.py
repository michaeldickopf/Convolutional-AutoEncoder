# !E:\Dokumente\001_Studium\Bachelor\Autoencoder\nn_structure.py
import os
import pickle

import numpy as np
import torch
from torch import nn
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import natsort
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
import optuna
import torch.nn.functional as F
import time
import cv2

# 1) Get the data

# init
source_train = "E:\\Dokumente\\001_Studium\\Bachelor\\Programmieren\\train_data_144x128_GSD16\\"
source_test = "E:\\Dokumente\\001_Studium\\Bachelor\\Programmieren\\test_data_144x128_GSD16\\"
source_linux_test = "/home/mdickopf/Daten_neu/test_data_144x128_GSD16"
source_linux_train = "/home/mdickopf/Daten_neu/train_data_144x128_GSD16"
is_Linux = True if os.name == 'posix' else False


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


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


# Saving part
transform = transforms.Compose([
    transforms.ToTensor()])

dataset_train = CustomDataSet(source_linux_train if is_Linux else source_train, transform=transform)
dataset_test = CustomDataSet(source_linux_test if is_Linux else source_test, transform=transform)

# 2) Load the Data

# dataloader
train_loader = DataLoader(dataset_train,
                          batch_size=P.batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset_test,
                         batch_size=P.batch_size,
                         shuffle=False)

# check Dataloaders
print(f"Length of train dataloader: {len(train_loader)} batches of {P.batch_size}")
print(f"Length of test dataloader: {len(test_loader)} batches of {P.batch_size}")


# 3) Build/ load a baseline model (Computer Vision)

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
            nn.Linear(18432, 10920),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.enc_layers(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ch, self.h, self.w = P.input_shape

        self.linear = nn.Sequential(
            nn.Linear(10920, 18432),
            nn.ReLU(),
        )
        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(64,
                                                        16, 18))

        self.dec_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 32x16x18
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),  # Output: 16x32x36
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Output: 16x64x72
            nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=2, output_padding=1),  # Output: 3x128x144
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.unflatten(x)
        x = self.dec_layers(x)
        x = torch.tanh(x)
        return x


# model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder()
decoder = Decoder()
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]
P.optimizer = torch.optim.Adamax(params_to_optimize, lr=P.lr, weight_decay=P.weight_decay)

# Check if the GPU is available
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)


# Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader
    i = 0
    for image_batch in tqdm(dataloader):  # (200,3,192,222)
        if i > 0:
            break
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # print(encoded_data.shape)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())
        i += 1
    return np.mean(train_loss)


# Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        i = 0
        for image_batch in tqdm(dataloader):
            if i > 0:
                break
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
            i += 1
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


diz_loss = {'train_loss': [], 'val_loss': []}
for epoch in tqdm(range(P.num_epochs)):
    train_loss = train_epoch(encoder, decoder, device, train_loader, P.loss_fn, P.optimizer)
    val_loss = test_epoch(encoder, decoder, device, test_loader, P.loss_fn)
    print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, P.num_epochs, train_loss, val_loss))
    diz_loss['train_loss'].append(train_loss)
    diz_loss['val_loss'].append(val_loss)

test_epoch(encoder, decoder, device, test_loader, P.loss_fn).item()


# with open('loss.pkl', 'wb') as fp:
#    pickle.dump(diz_loss, fp)


def plot_original_vs_reconstructed(encoder, decoder, dataloader, device):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        start_encoding = time.time()
        images = next(iter(dataloader)).to(device)
        encoded_images = encoder(images)

        first_image = encoded_images[0].cpu() if encoded_images.is_cuda else encoded_images[0]
        first_image = first_image.reshape(3, 56, 65)
        print(f'encoded-image-shape: {first_image.shape}')
        first_image_np = np.transpose(first_image.numpy(), (1, 2, 0))
        np.save('encoded-image.npy', np.uint8(first_image_np))
        saved_encoding = time.time()
        time_spent = saved_encoding - start_encoding
        print(f'time spent for encoding and saving: {time_spent}')
        reconstructed_images = decoder(encoded_images)

        original_images = images.cpu().numpy() if images.is_cuda else images.numpy()
        print(f'original-image_shape: {original_images[0].shape}')

        reconstructed_first_image = reconstructed_images[0].cpu() if reconstructed_images.is_cuda else \
            reconstructed_images[0]
        np.save('reconstructed-image.npy', np.transpose(np.uint8(reconstructed_first_image.numpy()), (1, 2, 0)))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(np.transpose(original_images[0], (1, 2, 0)))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(np.transpose(reconstructed_first_image, (1, 2, 0)))
        axes[1].set_title('Reconstructed Image')
        axes[1].axis('off')
        plt.savefig(f'test-reconstructed-{0}.png')


plot_original_vs_reconstructed(encoder, decoder, test_loader, device)

# Plot losses
plt.figure(figsize=(10, 8))
plt.semilogy(diz_loss['train_loss'], label='Train')
plt.semilogy(diz_loss['val_loss'], label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
# plt.grid()
plt.legend()
# plt.title('loss')
plt.savefig('loss.png')
plt.show()

# Speichern des Modells
# torch.save(encoder.state_dict(), "encoder.pth")
# torch.save(decoder.state_dict(), "decoder.pth")
