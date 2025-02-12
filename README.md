# GNNMoE

This repository provides a reference implementation of GNNMoE as described in the paper:

> **Mixture of Experts Meets Decoupled Message Passing: Towards General and Adaptive Node Classification**<br>

Available at [link](https://arxiv.org/abs/2412.08193).

## 1 Overview of Model Architecture and Performance

![img.png](imgs/model.png)
![img_1.png](imgs/result.png)
## 2 Python environment setup with Conda
```bash
  conda env create -f requirement.yaml
```

## 3 Data Preparation
For all datasets, we employ 10 different random seeds to ensure consistent data splits. Each dataset is randomly divided into training, validation, and testing sets with a fixed ratio of 48%, 32%, and 20%, respectively.

## 4 Code Execution
### 4.1 File Structure
- data -- put data in this dir
- model -- MoE model
- main.py -- run this 
- utils.py -- utils

### 4.2 Code Execution
```bash
  python main.py -D computers -M MoE
```

### 4.3 Hyper-parameter Space
- learning rate: [0.005, 0.01, 0.05, 0.1]
- dropout rate: [0.3, 0.5, 0.7, 0.9]

## Reference
````bash
@inproceedings{GNNMoE,
  author={Chen, Xuanze and Zhou, Jiajun and Yu, Shanqing and Xuan, Qi},
  title={Mixture of Experts Meets Decoupled Message Passing: Towards General and Adaptive Node Classification}, 
  booktitle = {Companion Proceedings of the ACM Web Conference 2025},
  year={2025},
  series = {WWW '25},
  doi={10.1145/3701716.3715462}
}
````
