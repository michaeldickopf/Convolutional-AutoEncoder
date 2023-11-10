# !E:\Dokumente\001_Studium\Bachelor\Autoencoder\nn_structure.py
import os
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

# 1) Get the data

# init
source_train = "E:\\Dokumente\\001_Studium\\Bachelor\\Programmieren\\train_data_144x128_GSD16\\"
source_test = "E:\\Dokumente\\001_Studium\\Bachelor\\Programmieren\\test_data_144x128_GSD16\\"
source_linux_test = "/home/mdickopf/Daten_neu/test_data_144x128_GSD16"
source_linux_train = "/home/mdickopf/Daten_neu/Bachelorarbeit/train_data_144x128_GSD16"
test_image = "E:\\Dokumente\\001_Studium\\Bachelor\\Autoencoder_neu\\Autoencoder\\test_image\\"
test_image_linux = "/home/mdickopf/Daten_neu/test_image/"
is_Linux = False


class Parameters:
    def __init__(self):
        self.input_shape = (3, 128, 144)
        self.encoder_f = [64, 64, 64, 64]
        self.decoder_f = [64, 3]
        self.lat_space_size = 11060
        self.lr = 0.0017  # https://towardsdatascience.com/how-to-choose-the-optimal-learning-rate-for-neural-networks-362111c5c783
        self.batch_size = 200
        self.num_epochs = 1


P = Parameters()


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
        self.ch, self.h, self.w = P.input_shape
        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                      padding_mode='reflect'),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(True),
        )
        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        #  Fully Connected Layer
        self.linear = nn.Linear(18432, 11000)

    def forward(self, x):
        x = self.encoder_cnn(x)
        # print(f'x size CNN {x.shape}')
        x = self.flatten(x)
        # print(f'x size flatten {x.shape}')
        x = self.linear(x)
        # print(f'x size linear {x.shape}')
        return x


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.ch, self.h, self.w = P.input_shape
        self.linear = nn.Linear(11000, 18432)
        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(64,
                                                        16, 18))

        self.decoder_conv = nn.Sequential(
            # nn.PixelShuffle(2),
            # [200, 64, 16, 18] -> [200, 64, 32, 36]
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            # [200, 64, 32, 36] -> [200, 32, 64, 72]
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            # [200, 32, 64, 72] -> [200, 16, 128, 144]
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            # [200, 16, 128, 144] -> [200, 3, 128, 144]
            # nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.linear(x)
        # print(f'linear shape in decoder: {x.shape}')
        x = self.unflatten(x)
        # print(f'unflattened in decoder: {x.shape}')
        x = self.decoder_conv(x)
        # print(f'decoded shape: {x.shape}')
        x.sigmoid()
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
    # plot_ae_outputs(encoder,decoder,n=10)

test_epoch(encoder, decoder, device, test_loader, loss_fn).item()


def plot_original_reconstructed(dataloader, encod, decod):
    """
    Plots original and reconstructed images side by side.

    :param encod: Encoder instance
    :param decod: Decoder instance
    :param original_data: List of original images.
    :param reconstructed_data: List of reconstructed images.
    """
    num_images = 5  # len(original_data)
    examples = next(iter(dataloader))
    

    reconstructed_data = decod(encod(dataloader))

    # Set up the subplot grid
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(10, num_images * 5))

    for i, img in enumerate(examples):
        if i > num_images-1:
            break

        # Plot original image
        ax = axes[i, 0]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Original {i + 1}')

        # Plot reconstructed image
        ax = axes[i, 1]
        ax.imshow(reconstructed_data[i])
        ax.axis('off')
        ax.set_title(f'Reconstructed {i + 1}')

    plt.tight_layout()
    plt.show()


# Beispielverwendung:
plot_original_reconstructed(test_loader, encoder, decoder)

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
