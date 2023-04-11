import augmentation, main_teacher, torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from custom_dataset import train_loader,test_loader
import faster_rcnn

imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

min_size, max_size = 600, 1000

RPN_config = {'anchor_scale' : (128, 256, 512), 'anchor_aspect_ratio' : (0.5, 1.0, 2.0), 'downsample' : 32, 
              'in_channels' : 1024,
              'num_anchors' : 9,
              'bbox_reg_weights' : (1., 1., 1., 1.),
              'iou_positive_thresh' : 0.7,
              'iou_negative_high' : 0.3,
              'iou_negative_low' : 0,
              'batch_size_per_image' : 256, 'positive_fraction' : 0.5,
              'min_size' : 16, 'nms_thresh' : 0.7,
              'top_n_train' : 2000, 'top_n_test' : 300}

FastRCNN_Teacher_config = {'output_size' : 7, 'downsample' : 32,
                   'out_channels' : 4096, 'num_classes' : 21,
                   'bbox_reg_weights' : (10., 10., 5., 5.),
                   'iou_positive_thresh' : 0.5, 'iou_negative_high' : 0.5, 'iou_negative_low' : 0.1,
                   'batch_size_per_image' : 128, 'positive_fraction' : 0.25, 
                   'min_size' : 1, 'nms_thresh' : 0.3, 
                   'score_thresh' : 0.05, 'top_n' : 50}

TRAIN_config = {'epochs' : 100,
                'lr' : 0.001, 'momentum' : 0.9, 'weight_decay' : 0.0001,
                'milestones' : [8], 'clip' : 10,
                'gamma' : 0.1,
                'epoch_freq' : 1, 'print_freq' : 1,
                'save' : True, 'SAVE_PATH' : './'}

Teacher_TEST_config = {'num_classes' : 21, 'iou_thresh' : 0.5, 'use_07_metric' : True}
TEST_config = {'num_classes' : 21, 'iou_thresh' : 0.5, 'use_07_metric' : True}

DEMO_config = {'min_size' : min_size, 'mean' : imagenet_mean, 'std' : imagenet_std, 'score_thresh' : 0.7}

gpu_id = 0

batch_size = 1

data_dir = '../Data/'

Teacher_model = faster_rcnn.FasterRCNN(RPN_config, FastRCNN_Teacher_config, gpu_id)

FasterRCNN = main_teacher.FasterRCNN(RPN_config, FastRCNN_Teacher_config, TRAIN_config, Teacher_TEST_config, DEMO_config, gpu_id)
FasterRCNN.train(train_loader,test_loader)