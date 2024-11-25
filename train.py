import os
import torch
from torch.utils.data import DataLoader

from util import loss2str, timestamp, write_obj, plot_losses
from config import load_config
from dataset import Dataset
from model import SuperRes

import matplotlib.pyplot as plt

def stat(ldx):
    return '{} min/max=({:.5f}, {:.5f}) ({:.5f}, {:.5f})'.format(ldx.shape, ldx.min(), ldx.max(), ldx.mean(), ldx.std())

def train(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res_dir = config.result_dir

    # Output paths
    log_path = os.path.join(res_dir, 'log.txt')
    ckpt_dir = os.path.join(res_dir, 'train_ckpt')
    model_dir = os.path.join(res_dir, 'model')
    for d in [ckpt_dir, model_dir]:
        os.makedirs(d, exist_ok=True)

    # Init dataset
    dataset = Dataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)
    print('Number of training data: {}'.format(len(dataset)))

    # Init model
    model = SuperRes(is_train=True, config=config, lrestshape=dataset.lrestshape, hrestshape=dataset.hrestshape, hfaces=dataset.hfaces, device=device)
    model = model.to(device)
    model.train()

    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("n_trainable_params={:,}".format(n_trainable_params))

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
                lr = model.optimizer.param_groups[0]['lr']
                log_str = "{} iter:{} epoch:{} lr:{:.6f} | {}".format(timestamp(), n_iter, epoch, lr, loss2str(losses))
                print(log_str)
                with open(log_path, 'a+') as f:
                    f.write(f'{log_str}\n')
            
            # loss plot
            if n_iter % config.plot_iter == 0:
                save_path = os.path.join(res_dir, 'losses.jpg')
                plot_losses(save_path, n_iter, epoch, loss_dic)
                print('loss plot: {}'.format(save_path))

            # training checkpoint
            if n_iter % config.train_ckpt_iter == 0:
                ckpt_dir_epoch = os.path.join(ckpt_dir, '{:04d}'.format(n_iter))
                os.makedirs(ckpt_dir_epoch, exist_ok=True)
                for batch_idx in range(min(config.train_ckpt_n_max, len(model.frame))):
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
                print('training checkpoint: {}'.format(ckpt_dir_epoch))
            n_iter += 1

        # save model
        is_last_epoch = (epoch == n_epochs)
        if epoch > 0 and epoch % config.save_model_epoch == 0 or is_last_epoch:
            save_path = os.path.join(model_dir, 'model_latest.pth')
            model.save_networks(save_path)
            print('model saved: {}'.format(save_path))

        # update learning rate
        lr = model.optimizer.param_groups[0]['lr']
        if lr > 1e-6:
            model.update_learning_rate()

if __name__ == "__main__":
    parser = load_config('train')
    config = parser.parse_args()
    
    os.makedirs(config.result_dir, exist_ok=True)

    print("Start")
    train(config)
    print("Done")