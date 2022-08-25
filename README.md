# Yformer-Time-Series-Forecasting (ECML 2022)

Paper link: https://arxiv.org/abs/2110.08255

### Abstract

Time series data is ubiquitous in research as well as in a wide variety of industrial applications. Effectively analyzing the available historical data and providing insights into the far future allows us to make effective decisions. Recent research has witnessed the superior performance of transformer-based architectures, especially in the regime of far horizon time series forecasting. However, the current state of the art sparse Transformer architectures fail to couple down- and upsampling procedures to produce outputs in a similar resolution as the input. We propose the Yformer model, based on a novel Y-shaped encoder-decoder architecture that (1) uses direct connection from the downscaled encoder layer to the corresponding upsampled decoder layer in a U-Net inspired architecture, (2) Combines the downscaling/upsampling with sparse attention to capture long-range effects, and (3) stabilizes the encoder-decoder stacks with the addition of an auxiliary reconstruction loss. Extensive experiments have been conducted with relevant baselines on four benchmark datasets, demonstrating an average improvement of 19.82, 18.41 percentage MSE and 13.62, 11.85 percentage MAE in comparison to the current state of the art for the univariate and the multivariate settings respectively.

# Depencency
```
Python            3.8.12
numpy             1.19.2
pandas            1.2.4
scipy             1.6.3
torch             1.8.1
```

# Badges
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yformer-u-net-inspired-transformer-1/time-series-forecasting-on-etth2-720)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-720?p=yformer-u-net-inspired-transformer-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yformer-u-net-inspired-transformer-1/time-series-forecasting-on-etth2-24)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-24?p=yformer-u-net-inspired-transformer-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yformer-u-net-inspired-transformer-1/time-series-forecasting-on-etth1-720)](https://paperswithcode.com/sota/time-series-forecasting-on-etth1-720?p=yformer-u-net-inspired-transformer-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yformer-u-net-inspired-transformer-1/time-series-forecasting-on-etth2-168)](https://paperswithcode.com/sota/time-series-forecasting-on-etth2-168?p=yformer-u-net-inspired-transformer-1)

# Citation
```
@article{madhusudhanan2021yformer,
  title={Yformer: U-Net Inspired Transformer Architecture for Far Horizon Time Series Forecasting},
  author={Madhusudhanan, Kiran and Burchert, Johannes and Duong-Trung, Nghia and Born, Stefan and Schmidt-Thieme, Lars},
  journal={arXiv preprint arXiv:2110.08255},
  year={2021}
}
```

# Contact
If you have any questions please contact us by email - kiranmadhusud@ismll.de
