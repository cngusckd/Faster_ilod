import torch, torchvision
import augmentation, torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader

class VOC_Detection_forilod_teacher(torch.utils.data.Dataset):
    def __init__(self, root, year, image_set, download, transforms, use_diff):
        self.dataset = torchvision.datasets.VOCDetection(root, year, image_set, download)
        self.transforms = transforms
        self.use_diff = use_diff
        self.VOC_LABELS = ('__background__', # always index 0
                           'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
                           'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 
                           'train', 'tvmonitor')
        
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        labels, bboxs = [], []
        for info in target['annotation']['object']:
            if self.use_diff or (int(info['difficult']) == 0):
                if(info['name'] not in ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
                           'diningtable', 'dog', 'horse','motorbike','person']): # TEACHER : 16 cateogory ~person : 4473 ~pottedplant : 4585 / only person:2008
                    continue
                labels.append(self.VOC_LABELS.index(info['name']))
                # Make pixel indexes 0-based
                bboxs.append(torch.FloatTensor([float(info['bndbox']['xmin'])-1, float(info['bndbox']['ymin'])-1, 
                                                float(info['bndbox']['xmax'])-1, float(info['bndbox']['ymax'])-1]))
        if len(labels) == 0 :
            return [0],[0],[0]
        else :    
            labels, bboxs = torch.tensor(labels, dtype=int), torch.stack(bboxs, dim=0)
            if self.transforms: img, bboxs = self.transforms(img, bboxs)
            return img, labels, bboxs

    def __len__(self):
        return len(self.dataset)

imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

min_size, max_size = 600, 1000

data_dir = '../Data/'

batch_size = 1

train_transform = augmentation.Compose([augmentation.Resize(min_size, max_size),
                                        augmentation.Flip(), augmentation.ToTensor(),
                                        augmentation.Normalize(mean = imagenet_mean, std = imagenet_std)])

train_dataset_for_teacher = VOC_Detection_forilod_teacher(root=data_dir, year='2007', image_set='trainval', download=True, transforms=train_transform, use_diff=False)
test_dataset_for_teacher = VOC_Detection_forilod_teacher(root=data_dir, year='2007', image_set='test', download=True, transforms=train_transform, use_diff=False)

train_loader = DataLoader(train_dataset_for_teacher, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset_for_teacher, batch_size=1, shuffle=True, num_workers=0)