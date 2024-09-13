import os
import torch
from torch.utils.data import DataLoader

from util import loss2str, timestamp, write_obj, plot_losses
from config import load_config
from dataset import Dataset
from model import SuperRes


def train(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res_dir = config.result_dir

    # Output paths
    log_path = os.path.join(res_dir, 'log.txt')
    ckpt_dir = os.path.join(res_dir, 'train_ckpt')

    # Init dataset
    dataset = Dataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)
    print('Number of training data: {}'.format(len(dataset)))

    # Init model
    model = SuperRes(is_train=True, config=config, lrestshape=dataset.lrestshape, hrestshape=dataset.hrestshape, hfaces=dataset.hfaces, device=device)
    model = model.to(device)
    loss_dic = {}
    for loss_name in model.loss_names:
        loss_dic[loss_name] = []

    # Init optimizer
    n_epochs = config.n_epochs
    n_iter = 0
    for epoch in range(n_epochs+1):
        for batch_idx, data in enumerate(dataloader):
            model.set_input(data)
            model.optimize_parameters()

            losses = model.get_losses()
            for ln, l in losses.items():
                loss_dic[ln].append(l.item())

            # print
            if n_iter % config.print_iter == 0:
                log_str = "{} iter:{} epoch:{} | {}".format(timestamp(), n_iter, epoch+1, n_epochs, loss2str(losses))
                print(log_str)
                with open(log_path, 'a+') as f:
                    f.write(f'{log_str}\n')
            
            # loss plot
            if n_iter % config.plot_iter == 0:
                save_path = os.path.join(res_dir, 'losses.jpg')
                plot_losses(save_path, n_iter, epoch, loss_dic)

            # training checkpoint
            if n_iter % config.train_ckpt_iter == 0:
                ckpt_dir_epoch = os.path.join(ckpt_dir, '{:04d}'.format(n_iter))
                os.makedirs(ckpt_dir_epoch, exist_ok=True)
                for batch_idx in range(config.train_ckpt_n_max):
                    frame = model.frame[batch_idx]
                    hx_true = model.hx_true[batch_idx].detach().cpu().numpy()
                    hx_pred = model.hx_pred[batch_idx].detach().cpu().numpy()
                    lx = dataset.lrestshape.numpy() + model.ldx[batch_idx].detach().cpu().numpy()

                    save_path = os.path.join(ckpt_dir_epoch, '{:02d}_high_true.obj'.format(frame))
                    write_obj(save_path, hx_true, faces=dataset.hfaces)
                    save_path = os.path.join(ckpt_dir_epoch, '{:02d}_high_pred.obj'.format(frame))
                    write_obj(save_path, hx_pred, faces=dataset.hfaces)
                    save_path = os.path.join(ckpt_dir_epoch, '{:02d}_low.obj'.format(frame))
                    write_obj(save_path, lx, faces=dataset.lfaces)

            n_iter += 1

if __name__ == "__main__":
    parser = load_config('train')
    config = parser.parse_args()
    
    os.makedirs(config.result_dir, exist_ok=True)

    print("\nStart")
    train(config)
    print("Done")