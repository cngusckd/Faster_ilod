import torch, torchvision
from custom_dataset import train_dataset_for_teacher, train_loader

backbone = torchvision.models.resnet50(weights = "ResNet50_Weights.IMAGENET1K_V1")
backbone.cuda(0)

for i, (images, labels, bboxs) in enumerate(train_loader):
    x = images.cuda(0)
    x = x.cuda(0)
    x = backbone.conv1(x)
    print(x.shape)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)

    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)

    print(x.shape)
    break