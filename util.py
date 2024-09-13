import numpy as np
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