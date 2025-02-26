# Denoising Diffusion Recommender Model
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
cd DDRM_LightGCN
python train.py
```

Run the SGL-based DDRM on ML-1M dataset:
```bash
cd DDRM_SGL
python train.py
```

Run the MF-based DDRM on Yelp dataset:
```bash
cd DDRM_MF
python train_yelp.py
```

Run the LightGCN-based DDRM on Yelp dataset:
```bash
cd DDRM_LightGCN
python train_yelp.py
```

Run the SGL-based DDRM on Yelp dataset:
```bash
cd DDRM_SGL
python train_yelp.py
```

Run the MF-based DDRM on Amazon-book dataset:
```bash
cd DDRM_MF
python train_amazon.py
```

Run the LightGCN-based DDRM on Amazon-book dataset:
```bash
cd DDRM_LightGCN
python train_amazon.py
```

Run the SGL-based DDRM on Amazon-book dataset:
```bash
cd DDRM_SGL
python train_amazon.py
```