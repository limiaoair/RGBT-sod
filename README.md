# **A Query-guided Decoder Fusion Model for Specificity Learning in Multi-modal Salient Object Detection**


## Network Architecture
![fig1.png](figs/fig1.png)

## Results and Saliency maps
We perform quantitative comparisons and qualitative comparisons with 6 co-SOD
methods on 3 datasets.
![fig2.jpg](figs/fig2.png)
![fig3.jpg](figs/fig3.png)

### Prerequisites
- Python 3.6
- Pytorch 1.10.2
- Torchvision 0.11.3
- Numpy 1.19.2

  install SSM
   ```
  pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
  pip install packaging
  pip install timm==0.4.12
  pip install pytest chardet yacs termcolor
  pip install submitit tensorboardX
  pip install triton==2.0.0
  pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
  pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
  pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
   ```

### Datasets
 Put the [CoCo-SEG](https://drive.google.com/file/d/1GbA_WKvJm04Z1tR8pTSzBdYVQ75avg4f/view), [CoCA](http://zhaozhang.net/coca.html), [CoSOD3k](http://dpfan.net/CoSOD3K/) and [Cosal2015](https://drive.google.com/u/0/uc?id=1mmYpGx17t8WocdPcw2WKeuFpz6VHoZ6K&export=download) datasets to `PGFusion/data` as the following structure:
  ```
  PGFusion
     ├── other codes
     ├── ...
     │ 
     └── data
           
           ├── CoCo-SEG (CoCo-SEG's image files)
           ├── CoCA (CoCA's image files)
           ├── CoSOD3k (CoSOD3k's image files)
           └── Cosal2015 (Cosal2015's image files)
  ```


### Contact
Feel free to send e-mails to me (lmiao@tongji.edu.cn).

## Relevant Literature

```text
@misc{ruan2024vmunetvisionmambaunet,
      title={VM-UNet: Vision Mamba UNet for Medical Image Segmentation}, 
      author={Jiacheng Ruan and Jincheng Li and Suncheng Xiang},
      year={2024},
      eprint={2402.02491},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2402.02491}, 
}

@misc{yu2022democracydoesmattercomprehensive,
      title={Democracy Does Matter: Comprehensive Feature Mining for Co-Salient Object Detection}, 
      author={Siyue Yu and Jimin Xiao and Bingfeng Zhang and Eng Gee Lim},
      year={2022},
      eprint={2203.05787},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2203.05787}, 
}

...
```



# RGBT SOD
A query-guided decoder fusion model for specificity learning in RGB-T salient object detection
the code come soon

# Results

<p align="center">
    <img src="./result1.png"/> <br />
    <em> 
    Figure 1: Subjective comparison among different methods. 
    </em>
</p>

<p align="center">
    <img src="./result2.png"/> <br />
    <em> 
    Figure 2: samples. 
    </em>
</p>

<p align="center">
    <img src="./result3.png"/> <br />
    <em> 
    Figure 3: feature visualisation. 
    </em>
</p>
