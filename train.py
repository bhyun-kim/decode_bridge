import torch

from dataset import BridgeAccDataset
from scipy.signal import find_peaks
from scipy.fft import fft
import numpy as np
from transforms import RandomCrop1D, RandomDrop, RandomNoise, ToTensor

from torchvision import transforms
from torch.utils.data import DataLoader

from pytorch_metric_learning import losses as metric_learning_losses

from model import SearchNetwork, Network, parse_alphas
from operations import ZeroOperation, FactorizedReduction, SeparableOperation

import torch.nn as nn

from collections import namedtuple



def main():

    data_dir = r'G:\My Drive\03. Work\10. Programming\contrastive-learning-bridge\data\09.21.2023 brg-stiffness reduction'
    train_data = BridgeAccDataset(data_dir=data_dir, split='train')
    test_data = BridgeAccDataset(data_dir=data_dir, split='test')

    metadata = train_data.get_metadata(0)

    transform = transforms.Compose([
        RandomCrop1D(output_size=2048),
        RandomNoise(max_noise_level=0.1),
        RandomDrop(),
        ToTensor()
    ])

    # define dataset
    train_data = BridgeAccDataset(data_dir=data_dir,
                                split='train',
                                transform=transform,
                                sns_loc=1)
    test_data = BridgeAccDataset(data_dir=data_dir,
                                split='test',
                                transform=transform,
                                sns_loc=1)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


    # find unique peaks 
    signal = metadata['acc'][0]
    signal_length = len(signal)
    sample_rate = metadata['sample_rate']
    sample_spacing = 1 / sample_rate
    freq_range = np.linspace(0.0, 1.0/(2.0*sample_spacing), signal_length//2)

    signal_fft = fft(signal)
    peaks, _ = find_peaks(
        2.0/signal_length * np.abs(signal_fft[:signal_length//2]), 
        height=0.08, 
        distance=100)

    freq_peaks = freq_range[peaks]


    xf_peaks = np.array(freq_peaks)

    results = 128 / xf_peaks

    closest_odds = np.where(np.floor(results) % 2 == 1, np.floor(results), np.floor(results) + 1)
    closest_odds = np.where(np.abs(closest_odds - results) > np.abs(closest_odds - 2 - results), closest_odds + 2, closest_odds)

    unique_closest_odds = np.unique(closest_odds)

    # TODO: refactor if we need to find more than 3 smallest unique odds
    smallest_three_unique_odds = np.sort(unique_closest_odds)[:3]

    smallest_three_unique_odds = smallest_three_unique_odds.astype(int)
    smallest_three_unique_odds = smallest_three_unique_odds.tolist()

    smallest_three_unique_odds

    # create search space 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cell_peaks = smallest_three_unique_odds
    num_candidates = 2 + 4*len(cell_peaks)


    search_net = SearchNetwork(
        num_candidates=num_candidates,
        num_intermediates=4,
        stride=4,
        num_cells=8,
        init_channels=16,
        peaks=cell_peaks,
        depth_multiplier_at_reduction=2
    )
    search_net = search_net.to(device)

    optimizer = torch.optim.Adam(search_net.parameters(), lr=0.001)
    loss_func = metric_learning_losses.NTXentLoss()

    # search model 
    epochs = 20

    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):

            # get data
            output = data['measurement'].unsqueeze(1)
            label = data['label']

            # send data to device
            output = output.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            embeddings = search_net(output)
            loss = loss_func(embeddings, label)
            loss.backward()
            optimizer.step()

            # print loss
            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Step: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')



    # parse results
    PRIMITIVES = [
        'none',
        'skip_connect'
    ]
    for i in cell_peaks:
        PRIMITIVES.append(f'max_pool_{i}x{i}')
        PRIMITIVES.append(f'avg_pool_{i}x{i}')
        PRIMITIVES.append(f'sep_conv_{i}x{i}')
        PRIMITIVES.append(f'dil_conv_{i}x{i}')


    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

    normal_weights = search_net.alphas()[0].data.cpu().numpy()
    reduce_weights = search_net.alphas()[1].data.cpu().numpy()

    normal_gene = parse_alphas(normal_weights)
    reduce_gene = parse_alphas(reduce_weights)
    print(normal_gene)
    print(reduce_gene)
    DARTS_V1 = Genotype(normal=normal_gene, normal_concat=[2, 3, 4, 5], reduce=reduce_gene, reduce_concat=[2, 3, 4, 5])



    OPS = {
    'none' : lambda channels, stride, affine: ZeroOperation(stride=stride),
    'skip_connect' : lambda channels, stride, affine: nn.Identity() if stride == 1 else FactorizedReduction(channels, channels)
    }

    for peak in cell_peaks:
        padding = (peak-1)//2
        OPS[f'max_pool_{peak}x{peak}'] = lambda channels, stride, affine: nn.MaxPool1d(peak, stride=stride, padding=padding)
        OPS[f'avg_pool_{peak}x{peak}'] = lambda channels, stride, affine: nn.AvgPool1d(peak, stride=stride, padding=padding, count_include_pad=False)
        OPS[f'sep_conv_{peak}x{peak}'] = lambda channels, stride, affine: SeparableOperation(channels, channels, peak, stride=stride, padding=padding, repeat=True)
        OPS[f'dil_conv_{peak}x{peak}'] = lambda channels, stride, affine: SeparableOperation(channels, channels, peak, stride=stride, padding=padding, dilation=2)

    # rebuild model
    net = Network(init_channels=16, num_cells=8, genotype=DARTS_V1, stride=4)
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = metric_learning_losses.NTXentLoss()

    # train model

    epochs = 10

    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):

            # get data
            output = data['measurement'].unsqueeze(1)
            label = data['label']

            # send data to device
            output = output.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            embeddings = net(output)
            loss = loss_func(embeddings, label)
            loss.backward()
            optimizer.step()

            # print loss
            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Step: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')



if __name__ == '__main__':
    main()