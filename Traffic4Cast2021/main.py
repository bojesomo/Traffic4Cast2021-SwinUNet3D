import pathlib
import sys
import os

# module_dir = str(pathlib.Path(os.getcwd()).parent)
module_dir = str(pathlib.Path(os.getcwd()))
sys.path.append(module_dir)

import argparse
import warnings

import numpy as np
import pandas as pd
import time
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
import torch

from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.plugins.ddp_sequential_plugin import DDPSequentialPlugin
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin
from pytorch_lightning.loggers import CSVLogger

from Traffic4Cast2021.model_pl import Model, all_models

from Traffic4Cast2021.dataset import DataGenerator
from Traffic4Cast2021.utils import model_summary


class DataModule(pl.LightningDataModule):
    """ Class to handle training/validation splits in a single object
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_dims = None
        self.val_batch_size = None

    def setup(self):
        start_time = time.time()
        self.train = DataGenerator(stage='training', **vars(self.args))
        self.val = []
        self.predict = DataGenerator(stage='test', **vars(self.args))
        print(f"Datasets loaded in {time.time() - start_time} seconds")
        self.train_dims = self.train.__len__()

        accepted_batch_size = [1, 2, 5, 10, 25]
        batch_idx = [t >= self.args.batch_size for t in accepted_batch_size].index(True)
        self.val_batch_size = accepted_batch_size[batch_idx]

    def __load_dataloader(self, dataset, shuffle=False, batch_size=1, pin=True):
        dl = DataLoader(dataset,
                        batch_size=batch_size, num_workers=self.args.workers,
                        shuffle=shuffle, pin_memory=pin)
        # prefetch_factor=2,
        # persistent_workers=False)
        return dl

    def train_dataloader(self):
        ds = self.train
        return self.__load_dataloader(ds, shuffle=True, batch_size=self.args.batch_size, pin=True)

    def val_dataloader(self):
        # val_loader = self.__load_dataloader(self.val, shuffle=False, batch_size=self.val_batch_size, pin=True)
        # predict_loader = self.__load_dataloader(self.predict, shuffle=False, batch_size=self.val_batch_size, pin=True)
        predict_loader = DataLoader(self.predict,
                                    batch_size=self.val_batch_size, num_workers=2,
                                    shuffle=False, pin_memory=True)
        return predict_loader  # [val_loader, predict_loader]

    def test_dataloader(self):
        # predict_loader = self.__load_dataloader(self.predict, shuffle=False, pin=True)
        predict_loader = DataLoader(self.predict,
                                    batch_size=self.val_batch_size, num_workers=2,
                                    shuffle=False, pin_memory=True)
        return predict_loader

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--sampling-step', type=int, default=12,
                            help='incremental:1  jumps:>1, maximum better:24 (default: 1)')
        parser.add_argument('--collapse-time', type=bool, default=False, help='collapse time axis')
        parser.add_argument('--use_extra', type=bool, default=False, help='use extra training sample (Default: False)')
        parser.add_argument('--use_static', type=bool, default=False, help='use static variable (Default: True)')
        parser.add_argument('--use_time_slot', type=bool, default=False, help='use time slots (Default: True)')
        parser.add_argument('--augment_data', type=bool, default=False,
                            help='use data augmentation to reduce over-fitting')
        parser.add_argument('--n_frame_in', type=int, default=12, help='number of incoming frames')
        parser.add_argument('--n_frame_out', type=int, default=6, help='number of frames to predict')
        parser.add_argument('--times_out', type=str, default='5:10:15:30:45:60',
                            help='actual timing out in minutes sep by :')
        parser.add_argument('--n_channels', type=int, default=8, help='total number of channels per frame')
        parser.add_argument('--n_static', type=int, default=9, help='number of static channels')
        parser.add_argument('--n_time', type=int, default=2, help='number of time channels')
        return parser


def modify_options(options, n_params):
    filename = '_'.join(
        [f"{item}" for item in (options.net_type, options.blk_type,
                                int(n_params))])
    options.filename = options.name or filename  # to account for resuming from a previous state
    #options.filename = options.checkpoint.split(os.sep)[0] or filename  # to account for resuming from a previous state

    options.versiondir = os.path.join(options.log_dir, options.filename, options.time_code)
    #os.makedirs(options.versiondir, exist_ok=True)
    #readme_file = os.path.join(options.versiondir, 'options.csv')
    #args_dict = vars(argparse.Namespace(**{'modelname': options.filename, 'num_params': n_params}, **vars(options)))
    #args_df = pd.DataFrame([args_dict])
    #if os.path.exists(readme_file):
    #    args_df.to_csv(readme_file, mode='a', index=False, header=False)
    #else:
    #    args_df.to_csv(readme_file, mode='a', index=False)

    return options


def get_trainer(options):  # gpus, max_epochs=20):
    """ get the trainer, modify here it's options:
        - save_top_k
        - max_epochs
     """
    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = CSVLogger(save_dir=options.log_dir,
                       name=options.filename,
                       version=options.time_code,
                       )  # time_code)
    resume_from_checkpoint = None
    #if options.checkpoint:
    #    resume_from_checkpoint = options.checkpoint
    if options.name and options.time_code:
        resume_from_checkpoint = os.path.join(options.versiondir, 'checkpoints', 'last.ckpt')

    checkpoint_callback = ModelCheckpoint(monitor='train_loss_epoch', mode='min', save_top_k=10,
                                          save_last=True, verbose=False,
                                          filename='{epoch:02d}-{train_loss_epoch:.6f}')

    callbacks = [lr_monitor, checkpoint_callback]
    if options.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='train_loss_epoch',  # should be found in logs  # TODO - create validation set
            patience=20,  # 3,
            strict=False,  # will act as disabled if monitor not found
            verbose=False,
            mode='min'
        )

        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(gpus=options.gpus,
                         max_epochs=options.epochs,
                         progress_bar_refresh_rate=10,  # 80,
                         deterministic=True,
                         # accumulate_grad_batches=5,  # 10,  # to stylishly increase the batch size
                         gradient_clip_val=1,  # to clip gradient value and prevent exploding gradient
                         gradient_clip_algorithm='value',
                         stochastic_weight_avg=True,  # smooth loss to prevent local minimal
                         default_root_dir=os.path.dirname(options.log_dir),
                         # limit_train_batches=10, limit_val_batches=10, # limit_test_batches=10, fast_dev_run=True,
                         callbacks=callbacks,
                         profiler='simple',
                         sync_batchnorm=True,
                         num_sanity_val_steps=0,
                         accelerator='ddp',
                         logger=logger,
                         resume_from_checkpoint=resume_from_checkpoint,
                         num_nodes=options.nodes,
                         plugins=DDPPlugin(num_nodes=options.nodes, find_unused_parameters=False),
                         # plugins=DeepSpeedPlugin(stage=2),
                         # plugins=DeepSpeedPlugin(stage=3, cpu_offload=True, partition_activations=True),
                         precision=options.precision,
                         # move_metrics_to_cpu=True,  # to avoid metric related GPU  memory bottleneck
                         )

    return trainer


def do_test(trainer, model, test_data):
    print("-----------------")
    print("--- TEST MODE ---")
    print("-----------------")
    scores = trainer.test(model, test_dataloaders=test_data)


def train(options):
    """ main training/evaluation method
    """

    # some needed stuffs
    warnings.filterwarnings("ignore")

    pl.seed_everything(options.manual_seed, workers=True)
    torch.manual_seed(options.manual_seed)
    torch.cuda.manual_seed_all(options.manual_seed)

    data = DataModule(options)
    data.setup()

    # add other depending args
    options.train_dims = data.train_dims
    options.val_batch_size = data.val_batch_size
    options.in_channels = options.n_frame_in * options.n_channels
    if options.use_static:
        options.in_channels += options.n_static
    if options.use_time_slot:
        options.in_channels += options.n_time
    options.n_classes = options.n_frame_out * options.n_channels

    # let's load model for printing structure
    model = all_models[options.blk_type](**vars(options))

    model.eval()
    # print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #del model
    x_all = torch.rand(1, options.in_channels, 495, 436)
    if options.blk_type.lower().endswith('3d'):
        x_all = torch.rand(1, options.n_frame_in, options.n_channels, 495, 436)
    _ = model_summary(model, x_all, print_summary=True, max_depth=1)
    # _ = model_summary(model, x_all, print_summary=True, max_depth=0)
    del model, x_all
    # raise ValueError()  # stop here for now
    # ------------
    # trainer
    # ------------
    options = modify_options(options, n_params)
    trainer = get_trainer(options)

    # ------
    # Model
    # -----
    model = Model(options)
    # print(model)

    print(options)
    # ------------
    # train & final validation
    # ------------
    if options.mode == 'train':
        print("-----------------")
        print("-- TRAIN MODE ---")
        print("-----------------")
        trainer.fit(model, datamodule=data)
    else:
        print("-----------------")
        print("--- TEST MODE ---")
        print("-----------------")
        trainer.test(model, datamodule=data)


def add_main_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("-g", "--gpus", type=int, required=False, default=1,
                        help="specify number of gpu (default: 1)")
    parser.add_argument("-n", "--nodes", type=int, required=False, default=1,
                        help="number of nodes to use")
    parser.add_argument("-m", "--mode", type=str, required=False, default='train',
                        help="choose mode: train (default)  / val")
    parser.add_argument("-j", "--workers", type=int, required=False, default=8,
                        help="number of workers")
    parser.add_argument("--early-stopping", type=bool, required=False, default=False,
                        help="use early stopping")
    parser.add_argument('--precision', type=int, default=32, help='precision to use for training', choices=[16, 32])
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')

    parser.add_argument('--manual-seed', default=0, type=int, help='manual global seed')
    parser.add_argument('--log-dir', default='/l/proj/kuex0005/Alabi/logs/T4C', help='base directory to save logs')
    parser.add_argument('--model-dir', default='', help='base directory to save logs')
    parser.add_argument('--checkpoint', default='', #'real_swinunet3d_594086/20210804T163936/checkpoints/last.ckpt'
                        help='identifier for model if already exist')
    parser.add_argument('--name', default='', #'real_swinunet3d_594086', #'real_swinunet3d_583222',  # 'sedenion_resnet_2258470',
                        help='identifier for model if already exist')
    parser.add_argument('--time-code', default='', #20210804T163936', # '20210804T234548',  # '20210623T083738',
                    help='identifier for model if already exist')
    parser.add_argument('--memory_efficient', type=bool, default=True, help='memory_efficient')
    # parser.add_argument('--initial-epoch', type=int, default=0, help='number of epochs done')
    return parser


def get_time_code():
    time_now = [f"{'0' if len(x) < 2 else ''}{x}" for x in np.array(time.localtime(), dtype=str)][:6]
    if os.path.exists('t.npy'):
        time_before = np.load('t.npy')  # .astype(np.int)
        if abs(int(''.join(time_before)) - int(''.join(time_now))) < 90:
            time_now = time_before
        else:
            np.save('t.npy', time_now)
    else:
        np.save('t.npy', time_now)
    time_now = ''.join(time_now[:3]) + 'T' + ''.join(time_now[3:])
    return time_now


def main():
    parser = argparse.ArgumentParser(description="Traffic4Cast2021 Arguments")
    parser = add_main_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = DataModule.add_data_specific_args(parser)
    options = parser.parse_args()

    time_code = get_time_code()
    options.time_code = options.time_code or time_code  # to account for resuming from a previous state

    train(options)


if __name__ == "__main__":
    main()
