import torch, math, time, argparse, os
# from .dataset import *
import random, dataset
import numpy as np
from imagenet_c import corrupt
from tqdm import *
import PIL

parser.add_argument('--dataset',
                    default='cub',
                    help='Training dataset, e.g. cub, cars'
                    )
parser.add_argument('--batch-size', default=150, type=int,
                    dest='sz_batch',
                    help='Number of samples per batch.'
                    )
parser.add_argument('--gpu-id', default=0, type=int,
                    help='ID of GPU that is used for training.'
                    )
parser.add_argument('--workers', default=0, type=int,
                    dest='nb_workers',
                    help='Number of workers for dataloader.'
                    )
args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

os.chdir('../data/')
data_root = os.getcwd()
# Dataset Loader and Sampler

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
            if args.dataset == 'cub':
                img_root = ev_dataset.root + "/" + noise + "/s" + str(i) + "/images/" + ev_dataset.im_paths[index][83:]
            elif args.dataset == 'cars':
                img_root = ev_dataset.root + "/" + noise + "/s" + str(i) + "/car_ims/" + ev_dataset.im_paths[index][-10:]
            im.save(img_root)
