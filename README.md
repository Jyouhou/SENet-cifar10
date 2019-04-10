# SENet-cifar10
Machine Learning Homework

## inherited from senet.pytorch
* `python cifar.py` runs SE-ResNet20 with Cifar10 dataset. Expected accuracy: 93%


## Timeline:
1. [x] check and re-write code
2. [x] 这周：reimplement the 93% experiment (92.67) 
3. [ ] 下周：整理几个经典的classification paper，加东西


### tricks
1. [x] 初始化？ done
2. [x] 优化器？ done
3. [x] 数据增强？ done

### To-Do
1. [x] 周1：check paper，找出几个重要的点
2. [x] 周1：简单implement
3. [x] 周2：add bottleneck
3. [ ] 周3-4：认真研读resnet和senet，复现出paper的accuracy


## how to run
`rlaunch --cpu=8 --gpu=4 --memory=65536 -- python3 -i Cifar10.py`



## Performance
| experiment | network | Data Aug | Optim | Acc | code |
|:------:|:------------:|:------:|:------:|:------:|:------:|:------------:|
| 0404 | se20 | `pad=4, crop=32; horizontal flip` | `SGD(lr=0.1,m=0.9,wd=1e-4, bs=64)` | 92.67 | `rlaunch --cpu=8 --gpu=4 --memory=65536 -- python3 -i Cifar10.py` |
| 0409 | se110(no bottleneck) | `assorted` | `SGD(lr=1,m=0.9,wd=1e-4, bs=256)` | 93.78 | `rlaunch --cpu=8 --gpu=4 --memory=65536 -- python3 -i Cifar10.py --network=se_resnet110 --aug --batch_size=256` |