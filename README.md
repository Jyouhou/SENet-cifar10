# SENet-cifar10
Machine Learning Homework

## inherited from senet.pytorch
* `python cifar.py` runs SE-ResNet20 with Cifar10 dataset. Expected accuracy: 93%


## Timeline:
1. [x] check and re-write code
2. [x] 这周：reimplement the 93% experiment (92.67) 
3. [x] 下周：整理几个经典的classification paper，加东西


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
7. [x] 周6：实现cut-out
9. [x] 4.14-16：炼丹
10. [x] 4.15: 对cutout和dropout进行比较
10. [ ] 4.16：写报告
10. [ ] ddl 4.17

#### 实验
- dropout: 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9， `rlaunch --cpu=8 --gpu=2 --memory=65536 -- python3 Cifar10.py --epochs=256 --dropout=`
- cutout: 2, 4, 6, 8, 12, 16, 20， `rlaunch --cpu=8 --gpu=2 --memory=65536 -- python3 Cifar10.py --epochs=256 --cutout=`


## how to run
`rlaunch --cpu=8 --gpu=2 --memory=65536 -- python3 -i Cifar10.py`



## Performance
| experiment | network | Optim | Acc(+se +cutout=16) | Acc(+se) | Acc (my) | Acc(resnet paper) | code |
|:------:|:------------:|:------:|:------:|:------:|:------:|:------:|:------:|:------------:|
| 0410 | res20 | default | 93.49 (+2.24) | 92.15 (+0.90) | 92.08 (+0.83) | 91.25 | `id=` |
| 0410 | res32 | default | 94.20 (+1.71) | 92.96 (+0.47) | 92.55 (+0.06) | 92.49 | `id=` |
| 0410 | res44 | default | 94.55 (+1.79) | 93.53 (+0.70) | 92.76 (-0.07) | 92.83 | `id=` |
| 0410 | res56 | default | 95.15 (+2.12) | 94.02 (+0.99) | 93.62 (+0.59) | 93.03 | `id=` |
| 0411 | res110 | default | 95.63 (+2.24) |94.70 (+1.31) | 93.70 (+0.31) | 93.39 | `id=` |
| 0411 | res20 | bs=64 | 93.46 (+2.21) | 92.79 (+1.54) | 92.50 (+1.25) | 91.25 | `id=` |
| 0411 | res110 | bs=64 | 95.85 (+2.22) | 94.81 (+1.18) | 94.61 (+0.98) | (93.63) | `id=` |


| experiment | network | cutout | dropout | Acc |
|:------:|:------:|:------:|:------:|:------:|
| bl | res20 | - | - | 92.15 |
| -- | res20 | - | 0.1 | 92.35 (+ 0.20) |
| -- | res20 | - | 0.2 | 92.35 (+ 0.20) |
| -- | res20 | - | 0.4 | 92.03 (-0.12) |
| -- | res20 | - | 0.5 | 92.16 (+0.01) |
| -- | res20 | - | 0.6 | 92.15 (+ 0.00) |
| -- | res20 | - | 0.8 | 91.67 (-0.48) |
| -- | res20 | - | 0.9 | 89.69 (-2.46) |
| bl | res20 | - | - | 92.15 |
| -- | res20 | 2 | - | 92.14 (-0.01) |
| -- | res20 | 4 | - | 92.76 (+0.61) |
| -- | res20 | 6 | - | 92.37 (+0.22) |
| -- | res20 | 8 | - | 93.12 (+0.97) |
| -- | res20 | 12 | - | 93.12 (+0.97) |
| -- | res20 | 16 | - | 93.27 (+1.12) |
| -- | res20 | 20 | - | 93.05 (+0.90) |


- DA: `generic = pad=4, crop=32; horizontal flip`, `assorted = using--aug`
- optim: `default = SGD(lr=0.1,m=0.9,wd=1e-4, bs=128)`