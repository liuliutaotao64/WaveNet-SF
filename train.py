import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import MyDataSet
from utils import read_train_data, read_val_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate,seed_everything
import shutil
import matplotlib.pyplot as plt
from model import WaveNet_SF

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    seed_everything(args.seed)
    # tb_writer = SummaryWriter()

    train_images_path, train_images_label = read_train_data(args.train_data_path)
    val_images_path, val_images_label = read_val_data(args.val_data_path)

    img_size = 448
    data_transform = {
        "train": transforms.Compose([transforms.RandomRotation(degrees=15),
                                     transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.2)),
                                     transforms.ColorJitter(brightness=0.2),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.21, 0.21, 0.21], [0.16, 0.16, 0.16])]),
        "val": transforms.Compose([transforms.Resize(456),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.21, 0.21, 0.21], [0.16, 0.16, 0.16])])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=4,
                                             collate_fn=val_dataset.collate_fn)

    model =WaveNet_SF().to(device)

    if args.RESUME == False:
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)['state_dict']

            # Delete the weight of the relevant category
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            model.load_state_dict(weights_dict, strict=False)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # All weights except head are frozen
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)

    if args.optimizer == 1:
        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    if args.optimizer == 2:
        optimizer = optim.SGD(pg, lr=args.lr,weight_decay=args.wd,momentum=0.9)

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    start_epoch = 0

    if args.RESUME:
        path_checkpoint = args.checkpoint_path
        print("model continue train")
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_schedule'])

    epochs_list = []
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    learning_rate_list = []

    for epoch in range(start_epoch + 1, args.epochs + 1):

        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        epochs_list.append(epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        learning_rate_list.append(optimizer.param_groups[0]["lr"])




        if best_acc < val_acc:
            if not os.path.isdir(args.model_weights):
                os.makedirs(args.model_weights)
            torch.save(model.state_dict(),os.path.join(args.model_weights,"model_best.pth"))
            print("Saved epoch{} as new best model".format(epoch))
            best_acc = val_acc



        if epoch % 10 == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'lr_schedule': lr_scheduler.state_dict()
            }
            checkpoint_path = os.path.join(args.model_weights,'checkpoint')
            if not os.path.isdir(checkpoint_path):
                os.makedirs(checkpoint_path)
            torch.save(checkpoint, os.path.join(checkpoint_path, f'checkpoint.pth'))
            print('checkpot save')

        #add loss, acc and lr into tensorboard
        print("[epoch {}] accuracy: {}".format(epoch, round(val_acc, 5)))





    loss_png_path = os.path.join(args.model_weights, 'loss.png')
    arruracy_png_path = os.path.join(args.model_weights, 'arruracy.png')
    learning_rate_png_path = os.path.join(args.model_weights, 'learning_rate.png')
    # Plot Loss and Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs_list, train_loss_list, label='Train Loss', marker='o')
    plt.plot(epochs_list, val_loss_list, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_png_path)
    plt.close()


    plt.figure(figsize=(12, 6))
    plt.plot(epochs_list, train_acc_list, label='Train Accuracy', marker='o')
    plt.plot(epochs_list, val_acc_list, label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(arruracy_png_path)
    plt.close()


    plt.figure(figsize=(6, 4))
    plt.plot(epochs_list, learning_rate_list, label='Learning Rate', marker='o', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Over Epochs')
    plt.legend()

    # Save the learning rate plot
    plt.savefig(learning_rate_png_path)
    plt.close()

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=38, help='random seed')
    parser.add_argument('--optimizer', type=int, default=1,help='1:wadam,2:sgd')


    parser.add_argument('--train_data_path', type=str, default="")
    parser.add_argument('--val_data_path', type=str, default="")


    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--RESUME', type=bool, default= False

                        )

    parser.add_argument('--utils_path', type=str, default="")
    parser.add_argument('--checkpoint_path', type=str, default='',help='')
    parser.add_argument('--model_weights', type=str, default='',help = '')
    parser.add_argument('--model_path', type=str, default='',help = '')
    opt = parser.parse_args()

    main(opt)
