
# End-to-End Metric Learning From Corrupted Images Using Triplet Dimensionality Reduction Loss

## Requirements

- Python3
- PyTorch (> 1.0)
- NumPy
- tqdm
- Pytorch-Metric-Learning
- imagenet_c


## Datasets

1. Download four public benchmarks for deep metric learning
   - CUB-200-2011   
   - Cars-196 
2. Extract the tgz or zip file into `./data/` (Exceptionally, for Cars-196, put the files in a `./data/cars196`)
3. `python make_corrupt_dataset.py --dataset cub` or  `python make_corrupt_dataset.py --dataset cars`

## Training Embedding Network
### CUB-200-2011

- Train a embedding network of Inception-BN (d=512) using **TripletPCA loss**

```bash
python train.py --gpu-id 0 \
                --loss TripletPCA \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 128 \
                --lr 1e-4 \
                --dataset cub 
```

- Train a embedding network of googlenet (d=512) using **TripletPCA loss**

```bash
python train.py --gpu-id 0 \
                --loss TripletPCA \
                --model googlenet \
                --embedding-size 512 \
                --batch-size 128 \
                --lr 1e-4 \
                --dataset cub 
```

### Cars-196

- Train a embedding network of Inception-BN (d=512) using **TripletPCA loss**

```bash
python train.py --gpu-id 0 \
                --loss TripletPCA \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 128 \
                --lr 1e-4 \
                --dataset cars 
```

- Train a embedding network of googlenet (d=512) using **TripletPCA loss**

```bash
python train.py --gpu-id 0 \
                --loss TripletPCA \
                --model googlenet \
                --embedding-size 512 \
                --batch-size 128 \
                --lr 1e-4 \
                --dataset cars 
```

