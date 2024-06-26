import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# from torch.profiler import profile, record_function, ProfilerActivity

gDevice = "cuda:0" # "cpu"

# ======================================================================================= # 
# Parameters
M = 35637                     # Number of vertices on high-res surface mesh
N = 15872                     # Number of vertices on low-res volumetric mesh
gKnn = 5                      # Number of KNN neighbors in Feature Embedding modules
gK = 20                       # Number of local neighbors in Upsampling module
emb_feat_dims = [35, 64, 128] # Number of latent dimensions in Feature Embedding modules
pos_feat_dim = 32             # Dimension of positional encoding
w_up_feat_dim = 128           # Latent dimension of Upsampling module
n_upscale_layers = 8          # Number of layers in Upsampling module
decoder_dim = 256             # Latent dimension of Decoder module
# ======================================================================================= # 

class SuperRes(nn.Module):
    def __init__(self):
        super(SuperRes, self).__init__()

        self.hrestshape = torch.rand(1, M, 3, dtype=torch.float)
        self.lrestshape = torch.rand(1, N, 3, dtype=torch.float)

        self.net_pos_emb = PosEmb()
        self.net_feat_emb = FeatEmb()
        self.net_upscale = Upscale()
        self.net_decoder = Decoder()
        print("SuperRes init")

    def forward(self, lx, ldx):
        # position embedding
        pos_emb = self.net_pos_emb(ldx)

        # feature encoding
        feat_in = torch.cat((ldx, pos_emb), dim=-1)
        self.feat_outs = self.net_feat_emb(feat_in)
        self.z = torch.cat((feat_in, self.feat_outs), dim=-1)

        # upscale
        latent_code = self.net_upscale(self.z)

        # reconstruct
        hdx_pred = self.net_decoder(self.hrestshape.repeat(latent_code.shape[0], 1, 1), latent_code)
        return hdx_pred

class PosEmb(nn.Module):
    def __init__(self):
        super(PosEmb, self).__init__()
        self.pos_emb = self._define_pos_emb(N, pos_feat_dim)
        self.pos_emb.requires_grad = False
        
        print("  PosEmb init")
    def forward(self, x):
        B = x.shape[0]
        pos_emb = self.pos_emb.unsqueeze(0).repeat(B, 1, 1).to(gDevice)
        return pos_emb

    def _define_pos_emb(self, n_verts, emb_dim):
        pos = torch.arange(0, n_verts, dtype=torch.float) # same as vertex index
        pos = pos.unsqueeze(dim=1) # (n,) -> (n, 1)
        _2i = torch.arange(0, emb_dim, step=2, dtype=torch.float)

        pos_emb = torch.rand(n_verts, emb_dim) # e.g., (n, 32)
        pos_emb[:, 0::2] = torch.sin(pos / (10000 ** (_2i / emb_dim)))
        pos_emb[:, 1::2] = torch.cos(pos / (10000 ** (_2i / emb_dim)))

        return pos_emb

class FeatEmb(nn.Module):
    def __init__(self):
        super(FeatEmb, self).__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(EmbModule(index=0, in_dim=emb_feat_dims[0], out_dim=emb_feat_dims[1]))
        self.blocks.append(EmbModule(index=1, in_dim=emb_feat_dims[1], out_dim=emb_feat_dims[2]))

        self.drop_out = nn.Dropout(p=0.2)
        self.fc_initial = nn.Sequential(
            nn.Linear(3, pos_feat_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(pos_feat_dim, pos_feat_dim)
        )
        
        self.drop_out = nn.Dropout(p=0.2)
        print("  FeatEmb init")

    def forward(self, feat):
        outputs = []
        for block in self.blocks:
            feat = self.drop_out(block(feat))
            outputs.append(feat)
        out = torch.cat(outputs, dim=-1)
        return out

class EmbModule(nn.Module):
    def __init__(self, index, in_dim, out_dim):
        super(EmbModule, self).__init__()
        self.index = index

        self.tet2tet_dist_idx = torch.randint(0, N-1, (1, N, gKnn), dtype=torch.long).to(gDevice)
        self.tet2tet_dist_mtx = torch.rand((1, N, gKnn), dtype=torch.float).to(gDevice)

        # edge conv
        self.edge_conv = nn.Sequential(
            nn.Conv2d(2*in_dim, 2*out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(2*out_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2*out_dim, out_dim, kernel_size=1, bias=True)
        )
        self.pairwise_distance = None

        self.fc_out = nn.Sequential(
            nn.Conv1d(3*out_dim+in_dim, 3*out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(3*out_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(3*out_dim, 2*out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(2*out_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(2*out_dim, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(out_dim, out_dim, kernel_size=1, stride=1, bias=True)
        )
        print("  EmbModule init")

    def forward(self, feat):
        B, n, feat_dim = feat.shape
        # ================================================ #
        # edge conv
        if self.index == 0:
            feat1, _ = get_graph_feature(feat.transpose(1, 2), k=gKnn, idx=self.tet2tet_dist_idx.repeat((B, 1, 1)), return_dist=False)
        else:
            feat1, _ = get_graph_feature(feat.transpose(1, 2), k=gKnn, idx=None, return_dist=False)
        feat1 = self.edge_conv(feat1)
        feat1 = feat1.max(-1, keepdim=False)[0] # (B, n, f)

        # global pooling
        feat1_max = F.adaptive_max_pool1d(feat1, 1)
        feat1_avg = F.adaptive_avg_pool1d(feat1, 1)

        # reshape to make feature dimension go last
        feat1 = feat1.transpose(1, 2)
        feat1_max = feat1_max.transpose(1, 2).repeat(1, n, 1)
        feat1_avg = feat1_avg.transpose(1, 2).repeat(1, n, 1)

        feat_out = torch.cat((feat, feat1, feat1_max, feat1_avg), dim=-1)
        feat_out = self.fc_out(feat_out.transpose(1,2)).transpose(1,2)
        
        return feat_out # (B, n, feat_dim) = same as input

class Upscale(nn.Module):
    def __init__(self):
        super(Upscale, self).__init__()

        hx_pos_centers = torch.rand(1, M, gK, 3, dtype=torch.float)
        lx_pos_grouped = torch.rand(1, M, gK, 3, dtype=torch.float)
        dists = torch.rand(1, M, gK, 1, dtype=torch.float)
        self.idx = torch.randint(0, N-1, (1, M, gK), dtype=torch.long)
        self.w_up_inputs = torch.cat((hx_pos_centers, lx_pos_grouped, dists), dim=-1)
        self.w_up_inputs = self.w_up_inputs.permute(0, 3, 1, 2).to(gDevice)
        self.w_up_inputs.requires_grad = False

        in_dim = 7
        conv_weights = []
        for _ in range(n_upscale_layers):
            conv_weights.extend([
                nn.Conv2d(in_dim, w_up_feat_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(w_up_feat_dim),
                Sine(),
            ])
            in_dim = w_up_feat_dim
        conv_weights.extend([nn.Conv2d(w_up_feat_dim, 1, kernel_size=1, stride=1, padding=0)])
        self.conv_weights = nn.Sequential(*conv_weights)
        print("  Upscale init")

    def forward(self, x_feat):
        B, n_ldx, feat_dim = x_feat.shape

        query_feat = self.w_up_inputs[0:B, :, :, :]
        self.w = torch.softmax(self.conv_weights(query_feat).permute(0, 2, 3, 1), dim=2) 
        x_feat_grouped = self.index_points(x_feat, self.idx)
        feat_H = torch.sum(self.w*x_feat_grouped, dim=2)

        return feat_H
    
    def index_points(self, points, indices):
        device = points.device
        b, m, k = indices.shape

        batch_indices = torch.arange(b, device=device).view(b, 1, 1).expand(-1, m, k)
        res = points[batch_indices, indices, :]
        return res
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # compute input dim
        h_dim = decoder_dim
        latent_dim = sum(emb_feat_dims)
        dims = [latent_dim, h_dim, h_dim//2, h_dim//4, h_dim//8, h_dim//16, 3]
        self.decoder = SirenNet(
            dims=dims,
            final_activation=None,
            w0_initial=30.
        )
        print("  Decoder init")

    def forward(self, x, z):
        hdx = self.decoder(z)
        return hdx

# ==================================================================================== # 
def knn(x, k, return_dist=False):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1] # (batch_size, num_points, k)
    if return_dist:
        return idx, -pairwise_distance
    else:
        return idx, None

def get_graph_feature(x, k=20, idx=None, return_dist=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx, pairwise_distance = knn(x, k=k+1, return_dist=return_dist)   # (batch_size, num_points, k)
        idx = idx[:, :, 1:]
    else:
        pairwise_distance = None

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points
    idx = idx + idx_base

    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature, pairwise_distance

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.rand(dim_out, dim_in)
        bias = torch.rand(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.activation = Sine(w0) if activation is None else activation

        self.fc = nn.Linear(dim_in, dim_out)
        self.fc.weight = nn.Parameter(weight)
        self.fc.bias = nn.Parameter(bias) if use_bias else None

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        # out =  F.linear(x, self.weight, self.bias)
        out = self.fc(x)
        out = self.activation(out)
        return out

class SirenNet(nn.Module):
    def __init__(self, dims, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super().__init__()
        self.num_layers = len(dims)-2

        self.layers = nn.ModuleList([])
        for ind in range(self.num_layers+1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dims[ind]
            dim_out = dims[ind+1]
            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_out,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in=dims[ind], dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)

# ==================================================================================== # 
def run():
    """
    For further optimization using onnx: https://pytorch.org/docs/stable/onnx.html
    """
    
    # INIT MODEL
    model = SuperRes()
    model = model.to(gDevice)
    model.eval()
    
    for param in model.parameters():
        param.grad = None

    # INIT LOGGERS
    batch_size = 1
    repetitions = 5
    timings=np.zeros((repetitions,1))

    # INIT DATA
    lx_restshape = torch.rand(batch_size, N, 3, dtype=torch.float).to(gDevice) # vertices of low-res simulation mesh in restshape
    ldx = torch.rand(batch_size, N, 3, dtype=torch.float).to(gDevice) # per-vertex displacements from restshape

    # GPU WARMUP
    for _ in range(3):
        model(lx_restshape, ldx)

    # RUN FORWARD
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

            starter.record()
            # ---------------------------------- #
            hdx_pred = model(lx_restshape, ldx) # Final high-res surface is obtained by: hx = hx_restshape + hdx_pred
            # ---------------------------------- #
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    print("--------------------------------------")
    print("iterations = {}\naverage = {:.03f} ms".format(repetitions, mean_syn))
    print("--------------------------------------")

if __name__ == "__main__":
    print("\nStart")
    run()
    print("Done")