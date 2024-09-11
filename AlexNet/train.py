import json
import os
import sys
import time

import torch
from datasets import tqdm
from torch import nn, optim
from torchvision import transforms, datasets

from AlexNet.model import AlexNet

# 如果有可使用GPU则使用GPU，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

data_root = os.path.abspath(os.path.join(os.getcwd(), '../'))
image_path = data_root + "/data_set/flower_data/"
print(image_path)

# 加载数据集
train_dataset = datasets.ImageFolder(root=image_path + "/train", transform=data_transforms['train'])
train_num = len(train_dataset)

# {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
flower_list = train_dataset.class_to_idx

cla_dict = dict((val, key) for key, val in flower_list.items())

# write dict into json file 数据持久化
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=nw)

validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transforms["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=4, shuffle=False,
                                              num_workers=nw)

print("using {} images for training, {} images for validation.".format(train_num,
                                                                       val_num))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "/val", transform=data_transforms['val'])
validate_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

net = AlexNet(num_classes=5, init_weights=True)
net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = './AlexNet.pth'
best_acc = 0.0
train_steps = len(train_loader)

for epoch in range(10):

    net.train(True)
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, 0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # 打印训练过程
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter() - t1)

    # validate
    # 只在训练过程dropout神经元，用eval()会关闭dropout
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
