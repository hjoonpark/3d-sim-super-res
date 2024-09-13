import torch
from torch.utils.data import DataLoader

from config import load_config
from dataset import Dataset
from model import SuperRes

def loss2str(loss_dic):
    out = ''
    for name, loss in loss_dic.items():
        out += '{}:{:.8f} '.format(name, loss)
    return out

def train(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # Init dataset
    dataset = Dataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)
    print('Number of training data: {}'.format(len(dataset)))

    # Init model
    model = SuperRes(is_train=True, config=config, lrestshape=dataset.lrestshape, hrestshape=dataset.hrestshape, hfaces=dataset.hfaces, device=device)
    model = model.to(device)

    # Init optimizer
    n_epochs = config.n_epochs
    for epoch in range(n_epochs):
        for batch_idx, data in enumerate(dataloader):
            model.set_input(data)
            model.optimize_parameters()

            loss_dic = model.get_losses()
            print(loss2str(loss_dic))

if __name__ == "__main__":
    parser = load_config('train')
    config = parser.parse_args()

    print("\nStart")
    train(config)
    print("Done")