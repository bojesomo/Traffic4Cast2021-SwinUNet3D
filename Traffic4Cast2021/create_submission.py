import os
from os.path import dirname, basename
import sys
sys.path.insert(0, dirname(os.getcwd()))

import datetime
import argparse
from Traffic4Cast2021.utils import write_dict, read_dict
import numpy as np
import torch
import zipfile
from pathlib import Path
import shutil


def write_results(root_folder, read_fn=None):
    if read_fn is None:
        # print(root_folder)
        path = str(list(Path(root_folder).rglob(f'**/**/*.*'))[0])
        ext = os.path.splitext(path)[-1]
        read_fn = {'.pt': torch.load, '.h5': read_dict}[ext]

    for folder in os.listdir(root_folder):
        path = os.path.join(root_folder, folder)
        print(os.path.join(*path.split(os.sep)[-2:]))
        final_name = os.path.join('_'.join(os.listdir(path)[0].split('_')[2:]))
        prediction = np.zeros((100, 6, 495, 436, 8), dtype=np.uint8)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            data = read_fn(file_path)
            prediction[data['index']] = data['pred']
            os.remove(file_path)
        final_path = os.path.join(path, final_name).split('.')[0] + '.h5'
        write_dict({'array': prediction}, final_path)
        del prediction


parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='/l/proj/kuex0005/Alabi/logs/T4C/',
                    help='root dir of the model to  submit')
parser.add_argument('--name', type=str, default='real_swinunet3d_486686',
                    help='name of the model to  submit')
parser.add_argument('--time_code', type=str, default='20210813T012534',
                    help='time_code as used in model folder')
# parser.add_argument('--version', type=str, default='20210621T082031', help='version_number as used in model folder')
parser.add_argument('--epoch', type=int, default=0, help='epoch number')

args = parser.parse_args()
time_spent = datetime.datetime.now()
submission_folder = os.path.join(args.log_dir, args.name, args.time_code if hasattr(args, 'time_code') else args.version,
                                 'inference', f'epoch={args.epoch}')

all_folders = [f.name for f in os.scandir(submission_folder) if f.is_dir()]
# for folder in os.listdir(submission_folder):
for folder in all_folders:
    print(f'working with {folder}')
    root_folder = os.path.join(submission_folder, folder)
    print('collating ...')
    write_results(root_folder)
    print('zipping ...')
    with zipfile.ZipFile(root_folder + '.zip', 'w') as z:
        files = list(Path(root_folder).rglob(f'**/**/*.h5'))
        for file in files:
            path = str(file)
            arcname = os.path.join(*path.split(os.sep)[-2:])
            z.write(path, arcname=arcname)
            print(arcname)
            os.remove(path)
    shutil.rmtree(root_folder)

print(f'done !!!   time spent == {datetime.datetime.now() - time_spent}')
