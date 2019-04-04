# SENet-cifar10
Machine Learning Homework

## inherited from senet.pytorch
* `python cifar.py` runs SE-ResNet20 with Cifar10 dataset. Expected accuracy: 93%


## Timeline:
1. [x] check and re-write code
2. [x] 这周：reimplement the 93% experiment (92.67) 
3. [ ] 下周：整理几个经典的classification paper，加东西


## how to run
`rlaunch --cpu=8 --gpu=4 --memory=65536 -- python3 -i Cifar10.py`



## Performance
| experiment | Data Aug | Optim | Acc | code |
|:------:|:------------:|:------:|:------:|:------------:|
| 0404 | `pad=4, crop=32; horizontal flip` | `SGD(lr=0.1,m=0.9,wd=1e-4, bs=64)` | 92.67 | `rlaunch --cpu=8 --gpu=4 --memory=65536 -- python3 -i Cifar10.py` |