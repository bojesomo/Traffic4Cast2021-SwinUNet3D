import os
import numpy as np
import torch
import re
from pytorch_model_summary import summary
import h5py


def write_dict(data_dict, file_path):
    """
    write data in gzipped h5 format.
    """

    with h5py.File(file_path, 'w', libver='latest') as f:
        for name, data_in in data_dict.items():
            f.create_dataset(name,
                             shape=data_in.shape,
                             data=data_in,
                             # chunks=(1, *data_in.shape[1:]),
                             compression='gzip',
                             dtype='uint8',
                             compression_opts=9)


def read_dict(file_path):
    with h5py.File(file_path if isinstance(file_path, str) else str(file_path), "r") as fr:
        data = {key: fr.get(key)[()] for key in fr.keys()}
        return data


def write_results(root_folder, read_fn):
    for folder in os.listdir(root_folder):
        path = os.path.join(root_folder, folder)
        final_name = os.path.join('_'.join(os.listdir(path)[0].split('_')[2:]))
        prediction = np.zeros((100, 6, 495, 436, 8), dtype=np.uint8)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            data = read_fn(file_path)
            prediction[data['index']] = data['pred']
            os.remove(file_path)
        final_path = os.path.join(path, final_name).split('.')[0] + '.ht'
        write_dict({'array': prediction}, final_path)
        del prediction


def model_summary(model, inputs, print_summary=False, max_depth=1, show_parent_layers=False):
    # _ = summary(model, x_in, print_summary=True)
    kwargs = {'max_depth': max_depth,
              'show_parent_layers': show_parent_layers}
    sT = summary(model, inputs, show_input=True, print_summary=False, **kwargs)
    sF = summary(model, inputs, show_input=False, print_summary=False, **kwargs)

    st = sT.split('\n')
    sf = sF.split('\n')

    sf1 = re.split(r'\s{2,}', sf[1])
    out_i = sf1.index('Output Shape')

    ss = []
    i_esc = []
    for i in range(0, len(st)):
        if len(re.split(r'\s{2,}', st[i])) == 1:
            ssi = st[i]
            if len(set(st[i])) == 1:
                i_esc.append(i)
        else:
            sfi = re.split(r'\s{2,}', sf[i])
            sti = re.split(r'\s{2,}', st[i])
            # ptr = st[i].index(sti[2]) + len(sti[2])
            # in_1 = sf[i].index(sfi[1]) + len(sfi[1])
            # in_2 = sf[i].index(sfi[2]) + len(sfi[2])

            ptr = st[i].index(sti[out_i]) + len(sti[out_i])
            in_1 = sf[i].index(sfi[out_i-1]) + len(sfi[out_i-1])
            in_2 = sf[i].index(sfi[out_i]) + len(sfi[out_i])
            ssi = st[i][:ptr] + sf[i][in_1:in_2] + st[i][ptr:]
        ss.append(ssi)

    n_str = max([len(s) for s in ss])
    for i in i_esc:
        ss[i] = ss[i][-1] * n_str

    ss = '\n'.join(ss)
    if print_summary:
        print(ss)

    return ss


