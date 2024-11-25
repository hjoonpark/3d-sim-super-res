import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_config(mode: str):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory')
    if mode == 'train':
        parser.add_argument('--flist_path', type=str, default='flist/train.txt', help='.txt file containing a list of data paths')
        parser.add_argument('--n_epochs', type=int, default=200, help='number of trainning epochs')
        parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
        parser.add_argument('--lr_decay_epoch', type=int, default=9999, help='learning rate decay frequency')
        parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help='learning rate decay step gamma')
        parser.add_argument('--w_recon', type=float, default=10.0, help='z regularization weight')
        parser.add_argument('--w_zreg', type=float, default=2.0, help='z regularization weight')
        parser.add_argument('--w_fn', type=float, default=1e-3, help='face normal loss weight')
        parser.add_argument('--print_iter', type=int, default=1, help='print frequency')
        parser.add_argument('--plot_iter', type=int, default=1000, help='print frequency')
        parser.add_argument('--train_ckpt_iter', type=int, default=1000, help='checkpoint frequency')
        parser.add_argument('--train_ckpt_n_max', type=int, default=3, help='number of training results to save')
        parser.add_argument('--save_model_epoch', type=int, default=100, help='model saving frequency')
        parser.add_argument('--shuffle', type=str2bool, nargs='?', const=True, default=True, help='whether to shuffle dataloader')
    elif mode == 'test':
        parser.add_argument('--flist_path', type=str, default='flist/test.txt', help='.txt file containing a list of data paths')
        parser.add_argument('--pretrained_path', type=str, default='data/pretrained/model_latest.pth', help='path to pretrained model weights')
    else:
        raise NotImplementedError('mode must be either "train" or "test".')
        
    parser.add_argument('--lrestshape_path', type=str, default='data/lrestshape.obj', help='path to low-res restshape mesh')
    parser.add_argument('--hrestshape_path', type=str, default='data/hrestshape.obj', help='path to high-res restshape mesh')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')

    # model
    parser.add_argument('--pos_feat_dim', type=int, default=32, help='dimension of positional encoding')
    parser.add_argument('--emb_feat_dims', type=str, default='35 64 128', help='latent dimensions of feature embeddings, separated by space')
    parser.add_argument('--knn', type=int, default=5, help='number of knn neighbors in feature embeddings')
    parser.add_argument('--tet2tet_dist_idx_path', type=str, default='data/geo_dist_tet2tet_idx.npy', help='geodesic distance among tet')
    parser.add_argument('--tet2tet_dist_path', type=str, default='data/geo_dist_tet2tet.npy', help='geodesic distance among tet')
    parser.add_argument('--surf2tet_dist_idx_path', type=str, default='data/geo_dist_surf2tet_idx.npy', help='geodesic distance between surf and tet')
    parser.add_argument('--surf2tet_dist_path', type=str, default='data/geo_dist_surf2tet.npy', help='geodesic distance between surf and tet')
    parser.add_argument('--upK', type=int, default=10, help='number of local neighbors in upsampling')
    parser.add_argument('--w_up_feat_dim', type=int, default=128, help='latent dimension of upsampling')
    parser.add_argument('--n_upscale_layers', type=int, default=8, help='number of layers in upsampling')
    parser.add_argument('--decoder_dim', type=int, default=256, help='latent dimension of decoder')
    parser.add_argument('--dropout', type=float, default=0, help='probability of an element to be zeroed')

    return parser