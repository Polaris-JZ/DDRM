# Denoising Diffusion Recommender Model
- last update : 2023-10-17 (update README.md)
This is the pytorch implementation of our paper
> Denoising Diffusion Recommender Model

## Environment
- Anaconda 3
- Python 3.8.12
- Pytorch 1.7.0
- Numpy 1.21.2

## Training
Run the MF-based DDRM on ML-1M dataset:
```bash
cd DDRM_MF
python train.py
```

Run the LightGCN-based DDRM on ML-1M dataset:
```bash
cd LightGCN_MF
python train.py
```

Run the SGL-based DDRM on ML-1M dataset:
```bash
cd SGL_MF
python train.py
```