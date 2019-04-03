import argparse
import time

import torch.optim as optim
from Training import Trainer, DataLoader
import senet
from torchvision.models import resnet


def main():
    loader = DataLoader()
    if args.network in dir(senet):
        model = getattr(senet, args.network)(num_classes=10)
    elif args.network in dir(resnet):
        model = getattr(resnet, args.network)(num_classes=10)
    else:
        raise ValueError('no such model')
    model.cuda()
    optimizer = optim.SGD(params=model.parameters(),
                          lr=args.lr,
                          momentum=args.m,
                          weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 80, 0.1)
    trainer = Trainer(model, optimizer, scheduler, args.GPU)
    his_max_acc = []
    for e in range(args.epochs):
        scheduler.step()
        t0 = time.time()
        loss, acc = trainer.train(loader.generator(True,
                                       args.batch_size,
                                       args.GPU))
        t1 = time.time()
        print(f'===== ===== Epoch {e+1}/{args.epochs} ===== =====')
        print(f'    train accuracy = {acc}, loss = {loss}, time lapse {t1-t0} seconds')

        t0 = time.time()
        acc = trainer.test(loader.generator(False,
                                       args.batch_size,
                                       args.GPU))
        his_max_acc.append(acc)
        t1 = time.time()
        print(f'    test accuracy = {acc}, best acc = {max(his_max_acc)}, time lapse {t1-t0} seconds')
    return his_max_acc



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--network", type=str, default='se_resnet18')
    parser.add_argument("--GPU", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--m", type=float, default=9e-1)
    parser.add_argument("--wd", type=float, default=1e-4)
    args = parser.parse_args()
    h_acc = main()
    with open('./result.txt', 'w')as f:
        print(','.join(map(str,h_acc)), file=f)
