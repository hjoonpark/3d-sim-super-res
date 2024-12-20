import os
import torch
import torch.utils.data as data
from util import read_obj

class Dataset(data.Dataset):
    def __init__(self, config):
        # load data paths
        self.lowres_paths, self.highres_paths = [], []
        with open(config.flist_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                l = line.strip().split(' ')
                self.lowres_paths.append(l[0])
                self.highres_paths.append(l[1])

        # load restshapes
        self.lrestshape, self.lfaces = read_obj(config.lrestshape_path)
        self.hrestshape, self.hfaces = read_obj(config.hrestshape_path)
        self.lrestshape = torch.from_numpy(self.lrestshape)
        self.hrestshape = torch.from_numpy(self.hrestshape)

        # (manual) normalization parameters
        ldx_min, ldx_max = -0.05092, 0.04998
        hdx_min, hdx_max = -0.04126, 0.0404
        ldx_std = 0.00460
        hdx_std = 0.00375

        self.mu = 0.5*(ldx_max + ldx_min)
        self.sigma = 0.5*(ldx_max - ldx_min)
        self.lrestshape = self.normalize(self.lrestshape, self.mu, self.sigma)
        self.hrestshape = self.normalize(self.hrestshape, self.mu, self.sigma)

        # basic caching
        self.cache = {}

    def normalize(self, x, mu, sigma):
        return (x-mu)/sigma
    
    def unnormalize(self, x, mu, sigma):
        return x*sigma + mu

    def __len__(self):
        return len(self.lowres_paths)

    def __getitem__(self, idx):
        lowres_path = self.lowres_paths[idx]
        highres_path = self.highres_paths[idx]

        # frame number
        frame = int(os.path.basename(lowres_path).split('.obj')[0])

        if lowres_path in self.cache:
            ldx, hdx = self.cache[lowres_path]
        else:
            # TODO - improve efficiency
            verts_low, _  = read_obj(lowres_path)
            verts_high, _ = read_obj(highres_path)
            verts_low  = torch.from_numpy(verts_low)
            verts_high = torch.from_numpy(verts_high)

            verts_low = self.normalize(verts_low, self.mu, self.sigma)
            verts_high = self.normalize(verts_high, self.mu, self.sigma)

            # compute displacements
            ldx = verts_low - self.lrestshape
            hdx = verts_high - self.hrestshape

            self.cache[lowres_path] = (ldx, hdx)

        return {
            'frame': frame,
            'ldx': ldx,
            'hdx': hdx,
        }