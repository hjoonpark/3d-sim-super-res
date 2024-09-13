import os
import torch
import torch.utils.data as data
from util import read_obj

class Dataset(data.Dataset):
    def __init__(self, config):
        # load data paths
        self.lowres_paths  = [path.strip() for path in open(config.flist_lowres_path,  'r').readlines()]
        self.highres_paths = [path.strip() for path in open(config.flist_highres_path, 'r').readlines()]
        assert len(self.lowres_paths) == len(self.highres_paths)

        # load restshapes
        self.lrestshape, self.lfaces = read_obj(config.lrestshape_path)
        self.hrestshape, self.hfaces = read_obj(config.hrestshape_path)
        self.lrestshape = torch.from_numpy(self.lrestshape)
        self.hrestshape = torch.from_numpy(self.hrestshape)

    def __len__(self):
        return len(self.lowres_paths)

    def __getitem__(self, idx):
        lowres_path = self.lowres_paths[idx]
        highres_path = self.highres_paths[idx]

        # frame number
        frame = int(os.path.basename(lowres_path).split('.obj')[0])

        # TODO - improve efficiency
        verts_low, _  = read_obj(lowres_path)
        verts_high, _ = read_obj(highres_path)
        verts_low  = torch.from_numpy(verts_low)
        verts_high = torch.from_numpy(verts_high)

        # compute displacements
        ldx = verts_low-self.lrestshape
        hdx = verts_high-self.hrestshape

        return {
            'frame': frame,
            'ldx': ldx,
            'hdx': hdx
        }