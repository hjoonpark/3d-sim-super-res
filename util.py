import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def read_obj(obj_path):
    v, f = [], []
    for line in open(obj_path, 'r').readlines():
        val = line.strip().split(" ")
        key = val[0].lower()
        if key == 'v':
            v1 = float(val[1])
            v2 = float(val[2])
            v3 = float(val[3])
            v.append([v1, v2, v3])
        elif key == 'f':
            f1 = int(val[1].split("//")[0])
            f2 = int(val[2].split("//")[0])
            f3 = int(val[3].split("//")[0])
            f.append([f1, f2, f3])
        else:
            pass

    return np.array(v).astype(np.float32), np.array(f)-1

def write_obj(save_path, points, vnormals=[], faces=[]):
    """
    x: (N, 3)
    faces: (M, 3) - triangle faces
    """
    with open(save_path, "w+") as file:
        for i, v in enumerate(points):
            file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            
        for i, vn in enumerate(vnormals):
            file.write("vn {} {} {}\n".format(vn[0], vn[1], vn[2]))
            
        for i, f in enumerate(faces):
            # i = face index
            # f: 3d vector
            f1 = f[0] + 1
            f2 = f[1] + 1
            f3 = f[2] + 1
            file.write("f {}//{} {}//{} {}//{}\n".format(f1, f1, f2, f2, f3, f3))

def loss2str(loss_dic):
    out = ''
    for name, loss in loss_dic.items():
        out += '{}:{:.8f} '.format(name, loss)
    return out

def timestamp():
    now = datetime.datetime.now()
    out = "[{}-{:02}-{:02} {:02}:{:02}:{:02}s]".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    return out

def plot_losses(save_path, n_iter, epoch, loss_dic):
    plt.figure(figsize=(10, 4))
    lws = []
    for k, l in loss_dic.items():
        if "total" in k.lower():
            lw = 1
            ls = '-'
        else:
            lw = 0.5 
            ls = '--'
        plt.plot(np.arange(len(l)), l, linewidth=lw, label="{}({:.04f})".format(k, l[-1]), linestyle=ls)
        lws.append(lw)
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    leg = plt.legend(loc='upper left')
    leg_lines = leg.get_lines()
    for i, lw in enumerate(lws):
        plt.setp(leg_lines[i], linewidth=lw*2)
    leg_texts = leg.get_texts()
    plt.setp(leg_texts, fontsize=12)
    plt.xlabel("n_iter")
    plt.yscale("log")
    plt.title("Losses\nn_iter {}, epoch {}".format(n_iter, epoch))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150)
    plt.close()