import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import tqdm
from torchvision import transforms, datasets

from model import vgg


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), '../'))

    image_path = data_root + '/data_set/flower_data/'

    train_dataset = datasets.ImageFolder(root=image_path + "train", transform=data_transforms['train'])
    train_num = len(train_dataset)

    # {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    flower_list = train_dataset.class_to_idx

    cla_dict = dict((val, key) for key, val in flower_list.items())

    json_str = json.dumps(cla_dict)
    with open('class_indices', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=4)

    val_dataset = datasets.ImageFolder(root=image_path + "val", transform=data_transforms['val'])
    val_num = len(val_dataset)
    validate_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=4)

    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=5, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 30
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
