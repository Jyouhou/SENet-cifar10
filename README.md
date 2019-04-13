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
3. [x] 周3-4：认真研读resnet和senet，复现出paper的accuracy
5. [x] 周五：可能要check下senet paper里的结果为什么这么好 -- > 新的resnet结构
6. [x] 周6：提升到paper水平，整理好实验
7. [ ] 周6：实现cut-out
8. [ ] 周7：写报告


## how to run
`rlaunch --cpu=8 --gpu=2 --memory=65536 -- python3 -i Cifar10.py`



## Performance
| experiment | network | Optim | Acc(+se +cutout) | Acc(+se) | Acc (my) | Acc(resnet paper) | code |
|:------:|:------------:|:------:|:------:|:------:|:------:|:------:|:------:|:------------:|
| 0410 | res20 | default | - | 92.15 (+0.90) | 92.08 (+0.83) | 91.25 | `id=` |
| 0410 | res32 | default | - | 92.96 (+0.47) | 92.55 (+0.06) | 92.49 | `id=` |
| 0410 | res44 | default | - | 93.53 (+0.70) | 92.76 (-0.07) | 92.83 | `id=` |
| 0410 | res56 | default | - | 94.02 (+0.99) | 93.62 (+0.59) | 93.03 | `id=` |
| 0411 | res110 | default | - |94.70 (+1.31) | 93.70 (+0.31) | 93.39 | `id=` |
| 0411 | res20 | bs=64 | - | 92.79 (+1.54) | 92.50 (+1.25) | 91.25 | `id=` |
| 0411 | res110 | bs=64 | - | 94.81 (+1.18) | 94.61 (+0.98) | (93.63) | `id=` |


- DA: `generic = pad=4, crop=32; horizontal flip`, `assorted = using--aug`
- optim: `default = SGD(lr=0.1,m=0.9,wd=1e-4, bs=128)`