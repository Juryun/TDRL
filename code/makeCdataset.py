import torch, math, time, argparse, os
# from .dataset import *
import random, dataset, utils, losses, net
import numpy as np

from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate
from imagenet_c import corrupt
from tqdm import *
import wandb
import PIL
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # set random seed for all gpus

parser = argparse.ArgumentParser(description=
                                 'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'
                                 + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`'
                                 )
# export directory, training and val datasets, test datasets
parser.add_argument('--LOG_DIR',
                    default='../logs',
                    help='Path to log folder'
                    )
parser.add_argument('--dataset',
                    default='cub',
                    help='Training dataset, e.g. cub, cars, SOP, Inshop'
                    )
parser.add_argument('--embedding-size', default=512, type=int,
                    dest='sz_embedding',
                    help='Size of embedding that is appended to backbone model.'
                    )
parser.add_argument('--batch-size', default=150, type=int,
                    dest='sz_batch',
                    help='Number of samples per batch.'
                    )
parser.add_argument('--epochs', default=60, type=int,
                    dest='nb_epochs',
                    help='Number of training epochs.'
                    )
parser.add_argument('--gpu-id', default=0, type=int,
                    help='ID of GPU that is used for training.'
                    )
parser.add_argument('--workers', default=0, type=int,
                    dest='nb_workers',
                    help='Number of workers for dataloader.'
                    )
parser.add_argument('--model', default='bn_inception',
                    help='Model for training'
                    )
parser.add_argument('--loss', default='Proxy_Anchor',
                    help='Criterion for training'
                    )
parser.add_argument('--optimizer', default='adamw',
                    help='Optimizer setting'
                    )
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Learning rate setting'
                    )
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    help='Weight decay setting'
                    )
parser.add_argument('--lr-decay-step', default=10, type=int,
                    help='Learning decay step setting'
                    )
parser.add_argument('--lr-decay-gamma', default=0.5, type=float,
                    help='Learning decay gamma setting'
                    )
parser.add_argument('--alpha', default=32, type=float,
                    help='Scaling Parameter setting'
                    )
parser.add_argument('--mrg', default=0.1, type=float,
                    help='Margin parameter setting'
                    )
parser.add_argument('--IPC', type=int,
                    help='Balanced sampling, images per class'
                    )
parser.add_argument('--warm', default=1, type=int,
                    help='Warmup training epochs'
                    )
parser.add_argument('--bn-freeze', default=1, type=int,
                    help='Batch normalization parameter freeze'
                    )
parser.add_argument('--l2-norm', default=1, type=int,
                    help='L2 normlization'
                    )
parser.add_argument('--remark', default='',
                    help='Any reamrk'
                    )

args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Directory for Log
LOG_DIR = args.LOG_DIR + '/logs_{}/{}_{}_embedding{}_alpha{}_mrg{}_{}_lr{}_batch{}{}'.format(args.dataset, args.model,
                                                                                             args.loss,
                                                                                             args.sz_embedding,
                                                                                             args.alpha,
                                                                                             args.mrg, args.optimizer,
                                                                                             args.lr, args.sz_batch,
                                                                                             args.remark)
# Wandb Initialization
wandb.init(project=args.dataset + '_ProxyAnchor', notes=LOG_DIR)
wandb.config.update(args)

os.chdir('../data/')
data_root = os.getcwd()
# Dataset Loader and Sampler
if args.dataset != 'Inshop':
    trn_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        mode='train',
        transform=dataset.utils.make_transform(
            is_train=True,
            is_inception=(args.model == 'bn_inception')
        ))
else:
    trn_dataset = Inshop_Dataset(
        root=data_root,
        mode='train',
        transform=dataset.utils.make_transform(
            is_train=True,
            is_inception=(args.model == 'bn_inception')
        ))

if args.IPC:
    balanced_sampler = sampler.BalancedSampler(trn_dataset, batch_size=args.sz_batch, images_per_class=args.IPC)
    batch_sampler = BatchSampler(balanced_sampler, batch_size=args.sz_batch, drop_last=True)
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        num_workers=args.nb_workers,
        pin_memory=True,
        batch_sampler=batch_sampler
    )
    print('Balanced Sampling')

else:
    dl_tr = torch.utils.data.DataLoader(
        trn_dataset,
        batch_size=args.sz_batch,
        shuffle=True,
        num_workers=args.nb_workers,
        drop_last=True,
        pin_memory=True
    )
    print('Random Sampling')

if args.dataset != 'Inshop':
    ev_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        mode='eval',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

else:
    query_dataset = Inshop_Dataset(
        root=data_root,
        mode='query',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

    gallery_dataset = Inshop_Dataset(
        root=data_root,
        mode='gallery',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        ))

    dl_gallery = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

nb_classes = trn_dataset.nb_classes()

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]
severity = 1
for noise in tqdm(CORRUPTIONS) :
    for i in range(1,6):
        for index in range(len(ev_dataset)):
            im = PIL.Image.open(ev_dataset.im_paths[index]).convert("RGB")

            im = im.resize((224,224))
            im = corrupt(np.array(im), corruption_name=noise, severity=i)
            im = PIL.Image.fromarray(im)
            thal = ev_dataset.root + "/" + noise + "/s" + str(i) + "/images/" + ev_dataset.im_paths[index][83:]
            im.save(thal)
