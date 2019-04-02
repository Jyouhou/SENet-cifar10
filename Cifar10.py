import argparse

import torch.optim as optim
from .Training import Trainer, DataLoader
from senet import se_resnet18
from torchvision.models import resnet18

network = {
    'se_resnet18':se_resnet18,
    'resnet18':resnet18
}


def main():
    loader = DataLoader()
    model = network[args.network](num_classes=10, reduction=args.reduction)
    optimizer = optim.SGD(params=model.parameters(),
                          lr=1e-1,
                          momentum=0.9,
                          weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 80, 0.1)
    trainer = Trainer(model, optimizer, scheduler, args.GPU)
    for e in range(args.epochs):
        scheduler.step()
        loss, acc = trainer.train(loader.generator(True,
                                       args.batch_size,
                                       args.GPU))
        print(f'===== ===== Epoch {e+1}/{args.epochs} ===== =====')
        print(f'    train accuracy = {acc}, loss = {loss}')
        acc = trainer.test(loader.generator(False,
                                       args.batch_size,
                                       args.GPU))
        print(f'    test accuracy = {acc}')


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--reduction", type=int, default=16)
    parser.add_argument("--network", type=str, default='se_resnet18')
    parser.add_argument("--GPU", type=int, default=4)
    args = parser.parse_args()
    main()
