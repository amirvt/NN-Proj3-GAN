import os
import pickle

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import torch
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# %%
# def mnist_data():
#     compose = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(
#             (.5, .5, .5),
#             (.5, .5, .5)
#         )
#     ])
#     return datasets.MNIST(root='./data', train=True, transform=compose, download=True)
#
#
# n_dim = 28
# n_features = 28 * 28
# n_dim_z = 100
# n_channels = 1
#
# batch_size = 128
# data = mnist_data()
# data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
# DATA_SET_NAME = 'MNIST'


# %%
def cifar_data():
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (.5, .5, .5),
            (.5, .5, .5)
        )
    ])
    return datasets.CIFAR10(root='./data', train=True, transform=compose, download=True)


batch_size = 128
data = cifar_data()
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

n_dim = 32
n_features = 32 ** 2
n_dim_z = 100
n_channels = 3
DATA_SET_NAME = 'CIFAR10'


# %%

def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0., 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1., 0.02)
        m.bias.data.fill_(0.)


class DiscriminatorDC(nn.Module):

    def __init__(self, convs):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, conv in enumerate(convs):
            self.layers.append(nn.Conv2d(*conv, bias=False))
            if i != 0 and i != len(convs) - 1:
                self.layers.append(nn.BatchNorm2d(conv[1]))
            if i != len(convs) - 1:
                self.layers.append(nn.LeakyReLU(0.2))
            else:
                self.layers.append(nn.Sigmoid())
        self.apply(init_weight)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


# 100 28 * 4 28 * 2 28
class GeneratorDC(nn.Module):

    def __init__(self, convs):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, conv in enumerate(convs):
            self.layers.append(nn.ConvTranspose2d(*conv, bias=False))
            if i != len(convs) - 1:
                self.layers.append(nn.BatchNorm2d(conv[1]))
                self.layers.append(nn.ReLU())
                # self.layers.append(nn.Dropout(0.25))
            else:
                self.layers.append(nn.Tanh())
        self.apply(init_weight)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class DiscriminatorMLP(nn.Module):

    def __init__(self, n_features):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features[i], n_features[i + 1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25)
            ) for i in range(len(n_features) - 1)]
        )

        self.layers.append(
            nn.Sequential(
                nn.Linear(n_features[-1], 1),
                nn.Sigmoid()
            )
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for l in self.layers:
            x = l(x)
        return x.view(x.shape[0], 1)


class GeneratorMLP(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_hiddens[i], n_hiddens[i + 1]),
                nn.LeakyReLU(0.2)
            ) for i in range(len(n_hiddens) - 2)])

        self.layers.append(
            nn.Sequential(
                nn.Linear(n_hiddens[-2], n_hiddens[-1]),
                nn.Tanh()
            )
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for l in self.layers:
            x = l(x)
        return x.view(x.shape[0], n_channels, n_dim, n_dim)


def noise(*dims):
    return Variable(torch.randn(dims, device=device))


def ones_target(n):
    return Variable(torch.ones(n, 1, device=device))


def zeros_target(n):
    return Variable(torch.zeros(n, 1, device=device))


# %% MNIST MLP
# MODEL_NAME = 'MLP'
# discriminator = DiscriminatorMLP([n_features, 512, 256]).to(device)
# generator = GeneratorMLP([n_dim_z, 256, 512, n_features]).to(device)
# %% MNIST DCGAN
# MODEL_NAME = 'DC'
# discriminator = DiscriminatorDC([(1, 32, 4, 2, 1), (32, 64, 4, 2, 1), (64, 1, 7, 1, 0)]).to(
#     device)  # 28 -> 14 -> 7 -> 1
# generator = GeneratorDC([(100, 64, 7, 1, 0), (64, 32, 4, 2, 1), (32, 1, 4, 2, 1)]).to(device)  # 7 -> 14 -> 28

# %% CIFAR10 DCGAN
MODEL_NAME = 'DC'
discriminator = DiscriminatorDC([
    (3, 64, 4, 2, 1),  # 16x16
    (64, 128, 4, 2, 1),  # 8x8
    (128, 256, 4, 2, 1),  # 4x4
    (256, 1, 4, 1, 0)  # 1x1
]).to(device)
generator = GeneratorDC([
    (100, 256, 4, 1, 0),  # 4x4
    (256, 128, 4, 2, 1),  # 8x8
    (128, 64, 4, 2, 1),  # 16x16
    (64, 3, 4, 2, 1)  # 32x32
]).to(device)

# %%
# discriminator = DiscriminatorMLP([n_features * 3, 1024, 512, 256]).to(device)
# generator = GeneratorMLP([n_dim_z, 256, 512, 1024, n_features * 3]).to(device)

# %%
optim_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss(reduction='sum')


def train_discriminator(real_data, fake_data):
    optim_d.zero_grad()
    n = real_data.shape[0]

    pred_real = discriminator(real_data)
    loss_real = criterion(pred_real, ones_target(n))
    loss_real.backward()

    num_correct = sum(pred_real >= 0.5).item()

    pred_fake = discriminator(fake_data)
    loss_fake = criterion(pred_fake, zeros_target(n))
    loss_fake.backward()

    num_correct += sum(pred_fake < 0.5).item()

    optim_d.step()

    return loss_real.item() + loss_fake.item(), num_correct


def train_generator(fake_data):
    optim_g.zero_grad()

    n = fake_data.shape[0]

    pred = discriminator(fake_data)
    loss = criterion(pred, ones_target(n))
    loss.backward()

    num_fooled = sum(pred >= 0.5).item()

    optim_g.step()

    return loss.item(), num_fooled


# %%
test_noise = noise(64, n_dim_z, 1, 1)


# def im2vec(images):
#     return images.view(images.shape[0], n_features)
#
#
# def vec2im(vectors):
#     return vectors.view(vectors.shape[0], n_dim, n_dim)


def show_image(images, epoch, save=True):
    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(np.transpose(vutils.make_grid(images.detach(), padding=2, normalize=True).cpu(), (1, 2, 0)))
    if save:
        _directory = 'images-' + DATA_SET_NAME + '-' + MODEL_NAME
        if not os.path.exists(_directory):
            os.makedirs(_directory)
        plt.savefig(_directory + '/epoch-' + str(epoch) + '.png', bbox="tight")
    else:
        plt.show()
    plt.close(fig)


def save_fake(epoch):
    fake = generator(test_noise)
    show_image(fake, epoch)


# %%
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


num_epochs = 200
hists = []
for i_epoch in range(1, num_epochs):
    loss_ds = 0
    loss_gs = 0
    num_corrects = 0
    num_fooleds = 0
    for n_batch, (real_batch, _) in enumerate(data_loader):
        N = real_batch.shape[0]

        # fix im2vec for mlp
        real_batch = Variable(real_batch.to(device))

        fake_batch = generator(noise(N, n_dim_z, 1, 1)).detach()

        loss_d, num_correct = train_discriminator(real_batch, fake_batch)

        fake_batch = generator(noise(N, n_dim_z, 1, 1))

        loss_g, num_fooled = train_generator(fake_batch)

        loss_ds += loss_d
        loss_gs += loss_g
        num_corrects += num_correct
        num_fooleds += num_fooled

    metrics = (
        np.round(loss_ds / (2 * len(data)), 4),
        np.round(num_corrects / (2 * len(data)), 3),
        np.round(loss_gs / len(data), 4),
        np.round(num_fooleds / len(data), 3)
    )

    hists.append(metrics)
    print(
        "[ Epoch: {:02d} ][ Loss_D: {:.4f}, Acc_D: {:.3f} ][ Loss_G: {:.3f}, ACC_G: {:.3f} ]".format(i_epoch, *metrics))
    save_fake(i_epoch)
    if i_epoch % 5 == 0:
        directory = 'images-' + DATA_SET_NAME + '-' + MODEL_NAME
        save_object(hists, directory + '/history.pkl')
        torch.save(generator.state_dict(), directory + '/gen-' + str(i_epoch))
        torch.save(discriminator.state_dict(), directory + '/dis-' + str(i_epoch))
