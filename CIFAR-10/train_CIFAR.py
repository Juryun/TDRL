from __future__ import print_function

import argparse
import os
import shutil
import time

import augmentations
from models.cifar.allconv import AllConvNet
import numpy as np
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnextriplet import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet
from losses import OnlineTripletPCALoss, OnlineTripletLoss
from datasets import BalancedBatchSampler, TripletAugCifar
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from utils import RandomNegativeTripletSelector, SemihardNegativeTripletSelector

parser = argparse.ArgumentParser(
    description='Trains AugMix + TripletPCA',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument(
    '--model',
    '-m',
    type=str,
    default='resnext',
    choices=['wrn', 'allconv', 'densenet', 'resnext'],
    help='Choose architecture.')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.001,
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0005,
    help='Weight decay (L2 penalty).')
# WRN Architecture options
parser.add_argument(
    '--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='Widen factor')
parser.add_argument(
    '--droprate', default=0.0, type=float, help='Dropout probability')
# AugMix options
parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument(
    '--aug-severity',
    default=3,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--no-jsd',
    '-nj',
    action='store_true',
    help='Turn off JSD consistency loss.')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all operations (+brightness,contrast,color,sharpness).')
# Checkpointing options
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default='./snapshots',
    help='Folder to save checkpoints.')
parser.add_argument(
    '--resume',
    '-r',
    type=str,
    default='',
    help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=50,
    help='Training loss print frequency (batches).')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=8,
    help='Number of pre-fetching threads.')

args = parser.parse_args()

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                               np.cos(step / total_steps * np.pi))


def aug(image, preprocess):
    """Perform AugMix augmentations and compute mixture.

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.

    Returns:
      mixed: Augmented and mixed image.
    """
    aug_list = augmentations.augmentations
    if args.all_ops:
        aug_list = augmentations.augmentations_all

    ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(args.mixture_width):
        image_aug = image.copy()
        depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
            1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, args.aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    # return mixed
    return preprocess(image)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                        aug(x, self.preprocess))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)


def train(net, train_loader, optimizer, scheduler):
    """Train for one epoch."""
    net.train()
    global count
    global kcount
    loss_ema = 0.
    margin = 10
    loss_fn = OnlineTripletPCALoss(margin, RandomNegativeTripletSelector(margin))
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        no_jsd = 0
        if no_jsd == 0:
            images = images.cuda()
            # data = tuple(d[0].cuda() for d in images)

            targets = targets.cuda()
            output = net(images)
            if type(output) not in (tuple, list):
                output = (output,)
            loss = loss_fn(*output, targets)
        else:
            # images_all = torch.cat(images, 0).cuda()
            # targets = targets.cuda()
            images_clean = tuple(d.cuda() for d in images[0])
            images_aug1 = tuple(d.cuda() for d in images[1])
            images_aug2 = tuple(d.cuda() for d in images[2])
            # data = tuple(d.cuda() for d in images_all)
            output_clean = net(*images_clean)
            output_aug1 = net(*images_aug1)
            output_aug2 = net(*images_aug2)
            if type(output_clean) not in (tuple, list):
                output_clean = (output_clean,)
            if type(output_aug1) not in (tuple, list):
                output_aug1 = (output_aug1,)
            if type(output_aug2) not in (tuple, list):
                output_aug2 = (output_aug2,)
            # logits_clean, logits_aug1, logits_aug2 = torch.split(
            #    output, images[0].size(0))

            # Cross-entropy is only computed on clean images
            # loss = F.cross_entropy(logits_clean, targets)

            p_clean, p_aug1, p_aug2 = loss_fn(*output_clean), loss_fn(*output_aug1), loss_fn(*output_aug2)

            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
        if i % args.print_freq == 0 and i != 0:
            print('Train Loss {:.3f}'.format(loss_ema))
            viz.line(Y=[loss_ema], X=np.array([count]), win=plot_loss, update='append')
            count = count + 1

    return loss_ema


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t, y in zip(T, Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()

    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in dataloader:
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training)  # revert to previous training state

    return [torch.stack(A[i]) for i in range(len(A))]


def evaluate_cos(model, dataloader):
    nb_classes = 10

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 1
    Y = []
    xs = []

    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y = Y.float().cpu()

    recall = []
    for k in [1]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall


def test(net, test_loader):
    """Evaluate network on given dataset."""
    net.eval()
    global account
    total_loss = 0.
    total_correct = 0
    count1 = 0
    with torch.no_grad():
        acc = evaluate_cos(net, test_loader)

    return total_loss / len(test_loader.dataset), acc[0]


def test_c(net, test_data, base_path):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        np_test_data = np.load(base_path + corruption + '.npy')
        np_test_labels = torch.LongTensor(np.load(base_path + 'labels.npy'))
        severity = []
        for i in range(0, 5):
            test_data.test_data = np_test_data[i * 10000:(i + 1) * 10000]
            test_data.test_labels = np_test_labels[i * 10000:(i + 1) * 10000]

            test_c_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True)

            test_c_loss, test_c_acc = test(net, test_c_loader)
            print(corruption + str(i) + " : " + str(100 - 100 * test_c_acc))
            severity.append(test_c_acc)
        sum_acc = np.mean(severity)
        corruption_accs.append(sum_acc)
        print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            corruption, test_c_loss, 100 - 100 * sum_acc))

    return np.mean(corruption_accs)


def test_cc(net, test_loader):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.
    total_correct = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0

    with torch.no_grad():
        test_embeddings = np.zeros((len(test_loader.dataset), 1024))
        test_labels = np.zeros(len(test_loader.dataset))
        k = 0
        for images, targets in test_loader:
            images = images.cuda()
            test_embeddings[k:k + len(images)] = net.get_embedding(images).data.cpu().numpy()
            test_labels[k:k + len(images)] = targets.numpy()
            k += len(images)

        emb1 = test_embeddings[0:10000]
        emb2 = test_embeddings[10000:20000]
        emb3 = test_embeddings[20000:30000]
        emb4 = test_embeddings[30000:40000]
        emb5 = test_embeddings[40000:50000]
        for i in range(len(emb1)):
            dist = torch.from_numpy(emb1 - emb1[i]).cuda().pow(2).sum(1)
            dist[i] = 999999
            values, indices = torch.topk(dist, 1, largest=False)
            for index in indices:
                if test_labels[i] == test_labels[index]:
                    count1 = count1 + 1
                    break
        acc1 = count1 / len(emb1)

        for i in range(len(emb2)):
            dist = torch.from_numpy(emb2 - emb2[i]).cuda().pow(2).sum(1)
            dist[i] = 999999
            values, indices = torch.topk(dist, 1, largest=False)
            for index in indices:
                if test_labels[i] == test_labels[index]:
                    count2 = count2 + 1
                    break
        acc2 = count2 / len(emb2)

        for i in range(len(emb3)):
            dist = torch.from_numpy(emb3 - emb3[i]).cuda().pow(2).sum(1)
            dist[i] = 999999
            values, indices = torch.topk(dist, 1, largest=False)
            for index in indices:
                if test_labels[i] == test_labels[index]:
                    count3 = count3 + 1
                    break
        acc3 = count3 / len(emb3)

        for i in range(len(emb4)):
            dist = torch.from_numpy(emb4 - emb4[i]).cuda().pow(2).sum(1)
            dist[i] = 999999
            values, indices = torch.topk(dist, 1, largest=False)
            for index in indices:
                if test_labels[i] == test_labels[index]:
                    count4 = count4 + 1
                    break
        acc4 = count4 / len(emb4)

        for i in range(len(emb5)):
            dist = torch.from_numpy(emb5 - emb5[i]).cuda().pow(2).sum(1)
            dist[i] = 999999
            values, indices = torch.topk(dist, 1, largest=False)
            for index in indices:
                if test_labels[i] == test_labels[index]:
                    count5 = count5 + 1
                    break
        acc5 = count5 / len(emb5)

        acc = acc1 + acc2 + acc3 + acc4 + acc5
        acc = acc / 5
        print("acc1 : " + str(acc1))
        print("acc2 : " + str(acc2))
        print("acc3 : " + str(acc3))
        print("acc4 : " + str(acc4))
        print("acc5 : " + str(acc5))
        # viz.line(Y=[acc], X=np.array([account]), win=plot_acc, update='append')
        # account = account+1

    return total_loss / len(test_loader.dataset), acc


def main():
    torch.manual_seed(1)
    np.random.seed(1)

    # Load datasets
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])
    process = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    preprocess = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.RandomHorizontalFlip(),  # 1
         transforms.RandomCrop(32, padding=4),  # 2
         transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])

    if args.dataset == 'cifar10':
        # train_data = datasets.CIFAR10(
        #    './data/cifar', train=True, transform=train_transform, download=True)
        train_data = datasets.CIFAR10(
            '/home/cvmlserver4/Juhyeon/repository/augmix/data/cifar', train=True, transform=train_transform,
            download=True)
        test_data = datasets.CIFAR10(
            '/home/cvmlserver4/Juhyeon/repository/augmix/data/cifar', train=False, transform=test_transform,
            download=True)
        base_c_path = '/home/cvmlserver4/Juhyeon/repository/augmix/data/cifar/CIFAR-10-C/'
        num_classes = 10
    else:
        train_data = datasets.CIFAR100(
            './data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(
            './data/cifar', train=False, transform=test_transform, download=True)
        base_c_path = './data/cifar/CIFAR-100-C/'
        num_classes = 100

    augdata = AugMixDataset(train_data, process, True)
    # for i in range(50000):
    #    train_data.train_data[i] = augdata[i][0]
    #  newdata = augdata[:]
    # train_data.train_data = augdata.train_data
    # train_data.transform = preprocess
    # triplet_train_data = TripletAugCifar(augdata)
    train_batch_sampler = BalancedBatchSampler(train_data.train_labels, n_classes=10, n_samples=20)

    train_loader = torch.utils.data.DataLoader(
        augdata,
        batch_sampler=train_batch_sampler,
        # shuffle=True,
        # batch_size= args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    # Create model
    if args.model == 'densenet':
        net = densenet(num_classes=num_classes)
    elif args.model == 'wrn':
        net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
    elif args.model == 'allconv':
        net = AllConvNet(num_classes)
    elif args.model == 'resnext':
        net = resnext29(num_classes=num_classes)

    net = net.cuda()

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.decay,
        nesterov=True)

    # Distribute model across all visible GPUs
    # net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Model restored from epoch:', start_epoch)

    if args.evaluate:
        # Evaluate clean accuracy first because test_c mutates underlying data
        test_loss, test_acc = test(net, test_loader)
        print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
            test_loss, 100 - 100. * test_acc))

        test_c_acc = test_c(net, test_data, base_c_path)
        print('Mean Corruption Error: {:.3f}'.format(100 - 10000. * test_c_acc))
        return

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            args.epochs * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)

    log_path = os.path.join(args.save,
                            args.dataset + '_' + args.model + '_training_log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

    best_acc = 0
    print('Beginning training from epoch:', start_epoch + 1)
    for epoch in range(start_epoch, args.epochs):
        begin_time = time.time()

        train_loss_ema = train(net, train_loader, optimizer, scheduler)
        test_loss, test_acc = test(net, test_loader)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        checkpoint = {
            'epoch': epoch,
            'dataset': args.dataset,
            'model': args.model,
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }

        save_path = os.path.join(args.save, 'checkpoint.pth.tar')
        torch.save(checkpoint, save_path)
        if is_best:
            shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

        with open(log_path, 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_time,
                train_loss_ema,
                test_loss,
                100 - 100. * test_acc,
            ))

        print(
            'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
            ' Test Error {4:.2f}'
                .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
                        test_loss, 100 - 100. * test_acc))

    test_c_acc = test_c(net, test_data, base_c_path)
    print('Mean Corruption Error: {:.3f}'.format(100 - 100 * test_c_acc))

    with open(log_path, 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
                (args.epochs + 1, 0, 0, 0, 100 - test_c_acc))


if __name__ == '__main__':
    main()
