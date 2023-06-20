# Knowledge Transfer-Driven Relation Complementation Network
The repo contains official PyTorch Implementation of paper [Knowledge Transfer-Driven Few-Shot Class-Incremental Learning]

### Overview
We present a Knowledge Transfer-Driven Relation Complementation Network (KT-RCNet) for few-shot class-incremental learning.
KT-RCNet consists of two main parts, a relation complementation strategy along with a knowledge transfer learning strategy named Random Episode Sampling and Augmentation (RESA). The relation complementation strategy employs a complementary model with a squared Euclidean-distance classifier as the auxiliary module to complement the results given by the widely used cosine classifier. RESA mimics the real incremental setting and constructs pseudo incremental tasks globally and locally to coincide with the learning objective of FSCIL and further improve the model's plasticity, respectively.

<img src='imgs/method.png' width='640' height='280'>


## Usage
### Installation
Option 1: 

``` pip install -r requirements.txt ```

Option 2: 
```
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
pip install tqdm
pip install numpy==1.24.3
```



## 
### 👉 Git the code
- ```git clone https://github.com/YeZiLaiXi/KT-RCNet.git```

### 👉 Prepare the data
- Download the whole datasets <datasets.tar.gz> from Baidu Netdisk to the folder KT-RCNet.

    Link：https://pan.baidu.com/s/1jBVGyR-L6gLHLvw-OBnoww 

    Code：z00o 

- Unpack <datasets.tar.gz>.

   ```tar -zxvf datasets.tar.gz```


### 👉 Prepare the model
- Download the pretrained models and our trained models <experiments.tar.gz> from Baidu Netdisk  to the folder KT-RCNet.

    Link：https://pan.baidu.com/s/13WtBhtvEnVRp5QJCCnjDWQ 

    Code：xmoo 

- Unpack <experiments.tar.gz>.

   ```tar -zxvf experiments.tar.gz```

<!-- ## 🌻 Inference or training -->
### 👉 Set the argument  ```train_flag``` to False in the corresponding script for inference (default) or True for model training
- miniImageNet：
  
    ```bash scripts/mini/mini.sh```

- CIFAR100:
  
    ```bash scripts/cifar/cifar.sh```

- CUB200：
  
    ```bash scripts/cub/cub.sh```


