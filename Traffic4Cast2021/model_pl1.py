import numpy as np
from torch.nn import functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torch
import os
from einops import rearrange

from argparse import ArgumentParser
from .models import (HyperSwinTransformer, HyperResNetUnet, HyperSwinEncoderDecoder,
                     HyperHybridSwinTransformer, HyperSwinEncoderDecoder3D, HyperDenseNetUnet,
                     HyperSwinUPerNet3D, HyperSwinUNet3D, HyperUNet)

from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

import tempfile
import zipfile
import torch_optimizer as extra_optim
from .dataset1 import T4CDataset
from .utils import write_dict, read_dict


optimizer_dict = {'adadelta': optim.Adadelta,
                  'adagrad': optim.Adagrad,
                  'adam': optim.Adam,
                  'adamw': optim.AdamW,
                  'sparse_adam': optim.SparseAdam,
                  'adamax': optim.Adamax,
                  'swats': extra_optim.SWATS,
                  'lamb': extra_optim.Lamb,
                  'asgd': optim.ASGD,
                  'sgd': optim.SGD,
                  'rprop': optim.Rprop,
                  'rmsprop': optim.RMSprop,
                  'lbfgs': optim.LBFGS}
all_models = {'resnet': HyperResNetUnet,
              'swin': HyperSwinTransformer,
              'hybridswin': HyperHybridSwinTransformer,
              'swinencoder': HyperSwinEncoderDecoder,
              'swinencoder3d': HyperSwinEncoderDecoder3D,
              'swinunet3d': HyperSwinUNet3D,
              'swinuper3d': HyperSwinUPerNet3D,
              'hyperunet': HyperUNet,
              'denseunet': HyperDenseNetUnet,
              # 'denseunet': HyperDenseUNet, 'denseunet3p': HyperDenseUNet
              }


class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()

        self.args = args

        self.net = all_models[args.blk_type](**vars(args))

        self.loss_fn = F.mse_loss
        self.num_gpus = self.args.gpus * self.args.nodes

        self.temp_dir = None
        self.zip_temporal = None
        self.zip_spatiotemporal = None
        # self.prediction = np.zeros((100, 6, 495, 436, 8)).astype(np.uint8)   # []
        self.count = 0
        self.filename = ''

        self.val_steps_per_file = math.ceil(100 /
                                            (self.args.val_batch_size * self.num_gpus))

    def forward(self, x, mask=None):
        out = self.net(x)
        if mask is not None:
            # print(out.shape, mask.shape)
            left_format = 'b h w'
            right_format = ' '.join(['b', ' '.join(['()']*(out.ndim - mask.ndim)), 'h w'])
            mask = rearrange(mask, f"{left_format} -> {right_format}")
            out = out * (~mask)
        return out

    def _compute_loss(self, y_hat, y, agg=True):
        if agg:
            loss = self.loss_fn(y_hat, y)
        else:
            loss = self.loss_fn(y_hat, y, reduction='none')
        return loss

    def process_batch(self, batch):
        in_out_seq, metadata = batch
        precision = self.args.precision
        return tuple([x.type(eval(f"torch.float{precision}")) / 255. for x in in_out_seq]), metadata

    def create_inference_dirs(self):
        for city in ['BERLIN', 'CHICAGO', 'ISTANBUL', 'MELBOURNE']:
            os.makedirs(os.path.join(self.temporal_dir, city), exist_ok=True)
        for city in ['NEWYORK', 'VIENNA']:
            os.makedirs(os.path.join(self.spatiotemporal_dir, city), exist_ok=True)

    def _on_epoch_start(self, stage='inference'):
        epoch_dir = os.path.join(self.args.versiondir, stage, f"epoch={self.current_epoch}")
        self.submission_dir = epoch_dir

        self.temporal_dir = os.path.join(epoch_dir, f'temporal_{self.current_epoch}')
        self.spatiotemporal_dir = os.path.join(epoch_dir, f'spatiotemporal_{self.current_epoch}')
        self.create_inference_dirs()

    def on_validation_epoch_start(self):
        self._on_epoch_start(stage='inference')

    def on_test_epoch_start(self):
        self._on_epoch_start(stage='test')

    def save_prediction(self, y_hat, metadata):
        filename = metadata['file_name'][0]
        city = filename.split('_')[0]
        if filename != self.filename:
            self.filename = filename
            # self.prediction = np.zeros((100, 6, 495, 436, 8)).astype(np.uint8)
            self.count = 0

        y_hat = T4CDataset.process_output(y_hat, collapse_time=self.args.collapse_time)
        self.count += 1  # y_hat.shape[0]

        if city in ['BERLIN', 'CHICAGO', 'ISTANBUL', 'MELBOURNE']:  # temporal
            # self.zip_temporal.write(temp_h5, arcname=arcname)
            file_path = os.path.join(self.temporal_dir, city,
                                     f"{str(self.device).split(':')[-1]}_{self.count}_" + os.path.splitext(filename)[0])
        else:  # spatiotemporal
            # self.zip_spatiotemporal.write(temp_h5, arcname=arcname)
            file_path = os.path.join(self.spatiotemporal_dir, city,
                                     f"{str(self.device).split(':')[-1]}_{self.count}_" +
                                     os.path.splitext(filename)[0])
        # print(f'saving ... {file_path}')
        torch.save({'pred': y_hat, 'index': metadata['part_index'].cpu().numpy()}, file_path + '.pt')
        # write_dict({'pred': y_hat, 'index': metadata['part_index'].cpu().numpy()}, file_path + '.h5')

    def training_step(self, batch, batch_idx, phase='train'):

        (x, y), metadata = self.process_batch(batch)
        # print(x.shape, y.shape, metadata)
        y_hat = self.forward(x, mask=metadata['mask'])
        # print(y_hat.shape)
        loss = self._compute_loss(y_hat, y)
        self.log(f'{phase}_loss', loss * 255**2, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, loader_idx, phase='val'):

        if loader_idx == 0:  # for validation loader only
            (x, y), metadata = self.process_batch(batch)
            y_hat = self.forward(x, mask=metadata['mask'])
            loss = self._compute_loss(y_hat, y)
            self.log(f'{phase}_loss', loss * 255**2, prog_bar=True, add_dataloader_idx=False)  # , logger=True)
        else:  # for prediction
            (x, ), metadata = self.process_batch(batch)
            y_hat = self.forward(x, mask=metadata['mask'])
            self.save_prediction(y_hat, metadata)

    def test_step(self, batch, batch_idx):  # , phase='test'):

        (x, ), metadata = self.process_batch(batch)
        y_hat = self.forward(x, mask=metadata['mask'])
        # y_hat = T4CDataset.process_output(y_hat.data.cpu())
        self.save_prediction(y_hat, metadata)

    @staticmethod
    def write_results(root_folder):
        for folder in os.listdir(root_folder):
            path = os.path.join(root_folder, folder)
            final_name = os.path.join('_'.join(os.listdir(path)[0].split('_')[2:]))
            prediction = np.zeros((100, 6, 495, 436, 8), dtype=np.uint8)
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                data = torch.load(file_path)
                # data = read_dict(file_path)
                prediction[data['index']] = data['pred']
                os.remove(file_path)
            final_path = os.path.join(path, final_name).split('.')[0] + '.ht'
            write_dict({'array': prediction}, final_path)
            del prediction

    def configure_optimizers(self):
        # print(self.args)
        if self.args.optimizer == 'sgd':
            other_args = {'lr': self.args.lr,
                          # 'weight_decay': self.args.weight_decay,
                          'nesterov': True,
                          'momentum': 0.9}
        elif self.args.optimizer == 'adam':
            other_args = {'lr': self.args.lr,
                          # 'weight_decay': self.args.weight_decay
                          }
        elif self.args.optimizer == 'lamb':
            other_args = {'lr': self.args.lr,
                          'clamp_value': 10,
                          # 'weight_decay': self.args.weight_decay
                          }
        elif self.args.optimizer == 'swats':
            other_args = {'lr': self.args.lr,
                          # 'weight_decay': self.args.weight_decay,
                          'nesterov': True,
                          # 'momentum': 0.9
                          }
        # create the optimizer
        no_decay = ['bias', 'absolute_pos_embed', 'relative_position_bias_table', 'norm']
        params_decay = [p for n, p in self.net.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.net.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.args.weight_decay},  # self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        # optimizer = optimizer_dict[self.args.optimizer](self.net.parameters(), **other_args)
        optimizer = optimizer_dict[self.args.optimizer](optim_groups, **other_args)

        if self.args.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, cooldown=0)  # , min_lr=1e-7)
            interval = 'epoch'
        elif self.args.scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr,
                                                            steps_per_epoch=math.ceil(
                                                                self.args.train_dims /
                                                                (self.args.batch_size * self.args.gpus
                                                                 * self.args.nodes)),
                                                            epochs=self.args.epochs, pct_start=0.1,
                                                            anneal_strategy='cos'  # 'linear
                                                            )
            interval = 'step'

        elif self.args.scheduler == 'cosine':
            t_max = math.ceil(self.args.epochs * self.args.train_dims /
                              (self.args.batch_size * self.args.gpus * self.args.nodes))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=1e-7)
            interval = 'step'

        scheduler = {'scheduler': scheduler,
                     'interval': interval,
                     'monitor': 'val_loss'}

        return {"optimizer": optimizer, "lr_scheduler": scheduler,
                "monitor": "val_loss"}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--use_group_norm', type=bool, default=False,
        #                     help='are we using group normalization [True, False]')
        parser.add_argument('--net_type', default='real', help='type of network',
                            choices=['sedenion', 'real', 'complex', 'quaternion', 'octonion'])
        parser.add_argument('--blk_type', default='swinunet3d', help='type of block',
                            choices=['resnet',
                                     'densenet', 'denseunet', 'denseunet3p', 'dpn',
                                     'swin', 'swinencoder', 'swinencoder3d', 'hybridswin',
                                     'swinunet3d', 'swinuper3d', 'hyperunet'])
        parser.add_argument('--dense_type', default='D', help='type of dense block in denseunet',
                             choices=['A', 'B', 'C', 'D'])
        parser.add_argument('--merge_type', default='both', help='type of decode block',
                            choices=['concat', 'add', 'both'])
        parser.add_argument('--use_neck', action='store_true',  # type=bool, default=False,
                            help='either use unet neck or not (default: False)')
        parser.add_argument('--use_se', action='store_true',  # type=bool, default=False,
                            help='either squeeze and excitation (default: False)')
        parser.add_argument('--constant_dim', action='store_true',  # type=bool, default=False,
                            help='are we using constant dim (default: False)')
        parser.add_argument('--mix_features', action='store_true',  # type=bool, default=False,
                            help='are we using mixing the features by default (default: False)')
        parser.add_argument('--patch_size', type=int, default=4, help='patch size to use in swin transformer')
        parser.add_argument('--depth', type=int, default=4, help='depth of encoder blocks (default: 1)')
        parser.add_argument('--decode_depth', type=int, default=None, help='depth of encoder blocks (default: None)')
        parser.add_argument('--mlp_ratio', type=int, default=4, help='mlp ratio of transformer layer (default: 4)')
        # parser.add_argument('--growth_rate', type=int, default=16 * 4, help='feature map per dense layer (default: 32)')
        # working on sf divisible by 16
        parser.add_argument('--start_filters', type=int, default=16 * 12,  # 16 * 8,
                            help='number of feature maps (default: 16*8)')
        parser.add_argument('--stages', type=int, default=4,
                            help='number of encoder stages (default:1)')
        parser.add_argument('--hidden_activation', default='silu', help='hidden layer activation')  # try gelu
        # parser.add_argument('--classifier_activation', default='sigmoid',
        #                     help='hidden layer activation (default: hardtanh)')  # sigmoid?
        parser.add_argument('--inplace_activation', type=bool, default=True, help='inplace activation')
        parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')

        parser.add_argument('--optimizer', default='adam', help='optimizer to train with (default: swats)',
                            choices=['sgd', 'adam', 'swats', 'lamb'])
        parser.add_argument('--scheduler', default='plateau', help='optimizer to train with',
                            choices=['plateau', 'onecycle', 'cosine'])

        parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='weight decay for regularization (default: 1e-2)')

        return parser

