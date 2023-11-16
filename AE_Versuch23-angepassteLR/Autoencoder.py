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
        self.encoder_f = [64, 64, 64, 64]
        self.decoder_f = [64, 3]
        self.lat_space_size = 11060
        self.lr = 0.0003724005169734309  # https://towardsdatascience.com/how-to-choose-the-optimal-learning-rate-for-neural-networks-362111c5c783
        self.batch_size = 200
        self.num_epochs = 10000


P = Parameters()

start = time.time()
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
stop = time.time()
res = stop-start
print(f'Time spent for loading Data: {res} seconds')
# check Dataloaders
print(f"Length of train dataloader: {len(train_loader)} batches of {P.batch_size}")
print(f"Length of test dataloader: {len(test_loader)} batches of {P.batch_size}")


# 3) Build/ load a baseline model (Computer Vision)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ch, self.h, self.w = P.input_shape  # Features: 55296

        # Convolutional section
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)  # Output: 16x64x72
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 16x32x36
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # Output: 32x16x18
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # Output: 64x8x9

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ch, self.h, self.w = P.input_shape

        # Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 8, 9))

        # Upsampling and Transpose Convolutional layers
        self.tconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # Output: 32x16x18
        self.tconv2 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)  # Output: 16x32x36
        # Output: 16x64x72
        self.tconv3 = nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=2, output_padding=1)  # Output: 3x128x144

    def forward(self, x):
        x = self.unflatten(x)
        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.tanh(self.tconv3(x))
        return x



# Define the loss function
loss_fn = torch.nn.MSELoss()

# Set the random seed for reproducible results
torch.manual_seed(0)

# model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder()
decoder = Decoder()
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=P.lr, weight_decay=1e-05)

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
    train_loss = train_epoch(encoder, decoder, device, train_loader, loss_fn, optim)
    val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)
    print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, P.num_epochs, train_loss, val_loss))
    diz_loss['train_loss'].append(train_loss)
    diz_loss['val_loss'].append(val_loss)

test_epoch(encoder, decoder, device, test_loader, loss_fn).item()
#with open('loss.pkl', 'wb') as fp:
#    pickle.dump(diz_loss, fp)


def plot_original_vs_reconstructed(encoder, decoder, dataloader, device):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for i, (images) in enumerate(dataloader):
            if i >= 1:
                break

            images = images.to(device)
            encoded_images = encoder(images)
            np_encoded_image = encoded_images[0].cpu().numpy() if encoded_images.is_cuda else encoded_images[0].numpy()
            print(f'encoded-image-shape: {np_encoded_image.shape}')
            print(f'encoded-image-size: {np_encoded_image.size}')
            np.save('encoded-image.npy', np_encoded_image)
            reconstructed_images = decoder(encoded_images)

            # Rescale images to [0, 1] range
            original_images = images.cpu().numpy() if images.is_cuda else images.numpy()
            print(f'original-image_shape: {original_images[0].shape}')
            reconstructed_images = reconstructed_images.cpu().numpy() if reconstructed_images.is_cuda else reconstructed_images.numpy()
            np.save('reconstructed-image.npy', reconstructed_images[0])

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].imshow(np.transpose(original_images[0], (1, 2, 0)))
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(np.transpose(reconstructed_images[0], (1, 2, 0)))
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
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(decoder.state_dict(), "decoder.pth")
