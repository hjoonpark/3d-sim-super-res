import os
import torch
from torch.utils.data import DataLoader

from util import loss2str, timestamp, write_obj, plot_losses
from config import load_config
from dataset import Dataset
from model import SuperRes

def test(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res_dir = config.result_dir

    # Output paths
    test_dir = os.path.join(res_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)

    # Init dataset
    dataset = Dataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    print('Number of test data: {}'.format(len(dataset)))

    # Init model
    model = SuperRes(is_train=False, config=config, lrestshape=dataset.lrestshape, hrestshape=dataset.hrestshape, hfaces=dataset.hfaces, device=device)
    model.load_networks(config.pretrained_path)
    model = model.to(device)
    model.eval()

    # Test
    lrestshape = dataset.unnormalize(model.lrestshape.detach().cpu().numpy(), dataset.mu, dataset.sigma)
    hrestshape = dataset.unnormalize(model.hrestshape.detach().cpu().numpy(), dataset.mu, dataset.sigma)
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            model.set_input(data)
            hdx_preds = model.forward(model.ldx)

            # dx to full shape
            hdx_preds = dataset.unnormalize(hdx_preds.detach().cpu().numpy(), dataset.mu, dataset.sigma)
            highres_pred = hrestshape + hdx_preds

            # save obj
            for batch_idx in range(hdx_preds.shape[0]):
                frame = model.frame[batch_idx]
                vertices = highres_pred[batch_idx]
                save_path = os.path.join(test_dir, '{:03d}.obj'.format(frame))
                write_obj(save_path, vertices, faces=model.hfaces)
                print("obj saved: {}".format(save_path))

if __name__ == "__main__":
    parser = load_config('test')
    config = parser.parse_args()
    
    os.makedirs(config.result_dir, exist_ok=True)

    print("Start")
    test(config)
    print("Done")