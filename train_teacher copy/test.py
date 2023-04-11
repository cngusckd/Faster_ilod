import torchsummary
import torch, torchvision
import torch.nn as nn
import rpn,fast_rcnn


# RPN
# first 'anchor_scale' : (128, 256, 512)
RPN_config = {'anchor_scale' : (128, 256, 512), 'anchor_aspect_ratio' : (0.5, 1.0, 2.0), 'downsample' : 16, 
              'in_channels' : 2048,
              'num_anchors' : 9,
              'bbox_reg_weights' : (1., 1., 1., 1.),
              'iou_positive_thresh' : 0.7,
              'iou_negative_high' : 0.3,
              'iou_negative_low' : 0,
              'batch_size_per_image' : 256, 'positive_fraction' : 0.5,
              'min_size' : 16, 'nms_thresh' : 0.7,
              'top_n_train' : 2000, 'top_n_test' : 300}


def build_RPN(self, RPN_config, gpu_id):
        anchor_generator = rpn.AnchorGenerator(RPN_config['anchor_scale'], RPN_config['anchor_aspect_ratio'], 
                                               RPN_config['downsample'], gpu_id)
        rpn_head = rpn.RPNHead(RPN_config['in_channels'], RPN_config['num_anchors'], gpu_id)
        RPN = rpn.RegionProposalNetwork(anchor_generator, rpn_head, 
                                        RPN_config['bbox_reg_weights'], 
                                        RPN_config['iou_positive_thresh'], 
                                        RPN_config['iou_negative_high'], RPN_config['iou_negative_low'],
                                        RPN_config['batch_size_per_image'], RPN_config['positive_fraction'], 
                                        RPN_config['min_size'], RPN_config['nms_thresh'], 
                                        RPN_config['top_n_train'], RPN_config['top_n_test'])
        return RPN


#####################################################################################

#Fast_rcnn

FastRCNN_config = {'output_size' : 7, 'downsample' : 16, 
                   'out_channels' : 4096, 'num_classes' : 21,
                   'bbox_reg_weights' : (10., 10., 5., 5.),
                   'iou_positive_thresh' : 0.5, 'iou_negative_high' : 0.5, 'iou_negative_low' : 0.1,
                   'batch_size_per_image' : 128, 'positive_fraction' : 0.25, 
                   'min_size' : 1, 'nms_thresh' : 0.3, 
                   'score_thresh' : 0.05, 'top_n' : 50}

def build_FastRCNN(self, FastRCNN_config, gpu_id):
        
        backbone_fc = nn.Sequential(
            nn.Linear(2048* 7* 7, 4096),
            nn.Linear(4096, 4096)
            ).cuda(gpu_id)
        #print(backbone_fc)

        roi_head = fast_rcnn.RoIHead(FastRCNN_config['output_size'], FastRCNN_config['downsample'], 
                                     backbone_fc, FastRCNN_config['out_channels'], FastRCNN_config['num_classes'], gpu_id)
        
        FastRCNN = fast_rcnn.FastRCNN(roi_head,
                                      FastRCNN_config['bbox_reg_weights'],
                                      FastRCNN_config['iou_positive_thresh'], 
                                      FastRCNN_config['iou_negative_high'], FastRCNN_config['iou_negative_low'],
                                      FastRCNN_config['batch_size_per_image'], FastRCNN_config['positive_fraction'],
                                      FastRCNN_config['min_size'], FastRCNN_config['nms_thresh'], 
                                      FastRCNN_config['score_thresh'], FastRCNN_config['top_n'])
        return FastRCNN


##########################################################################



backbone_resnet50 = torchvision.models.resnet50(weights = "ResNet50_Weights.IMAGENET1K_V1")
#torchsummary.summary(backbone_resnet50,(3,600,900))

backbone_resnet50 = nn.Sequential(*list(backbone_resnet50.children())[:-2])
#torchsummary.summary(backbone_resnet50,(3,600,900))

backbone_vgg16 = torchvision.models.vgg16(weights="VGG16_Weights.DEFAULT").features[:30]
#torchsummary.summary(backbone_vgg16,(3,600,800))



Fast_rcnn = build_FastRCNN(self = None, FastRCNN_config=FastRCNN_config, gpu_id=0 )

backbone_fc = nn.Sequential(nn.Linear(2048* 7* 7, 4096),nn.Linear(4096, 4096)).cuda(0)
classification = nn.Linear(4096, 21).cuda(0)
bbox_regressor = nn.Linear(4096, 4* 21).cuda(0)

from custom_dataset import train_dataset_for_teacher, train_loader

for i, (images, labels, bboxs) in enumerate(train_loader):
    if(images == [0]):
        continue
    labels, bboxs = labels.cuda(0), bboxs.cuda(0)
    x = images.cuda(0)
    input_x = x
    
    backbone_resnet50 = backbone_resnet50.cuda(0)
    x = backbone_resnet50(x)
    del backbone_resnet50
    torch.cuda.empty_cache()
    
    N, C, f_h, f_w = x.shape    
    print(f"입력 tensor = {input_x.shape}, backbone 통과 후 tensor = {x.shape}")
    
    del input_x
    torch.cuda.empty_cache()
    
    RPN = build_RPN(self = None, RPN_config = RPN_config, gpu_id=0)
    
    
    proposals, _, _ = RPN(images, x, labels, bboxs)
    del RPN
    torch.cuda.empty_cache()
    
    print(f"proposals shape : {proposals.shape}")
    
    proposals = proposals.detach()
    proposals_list = [proposal for proposal in proposals]
    
    bbox_features = torchvision.ops.roi_pool(x, proposals_list, 7, 1 / 32)
    bbox_features = bbox_features.view(N, -1, C, 7, 7)
    
    print(bbox_features.shape)
    
    bbox_features = torch.flatten(bbox_features, start_dim=2)
    bbox_features = backbone_fc(bbox_features)
    
    objectness = classification(bbox_features)
    pred_bbox_deltas = bbox_regressor(bbox_features)
    
    print(objectness.shape, pred_bbox_deltas.shape)
    
    #print(f"proposal shape : {proposals.shape}")
    _, _, _, _, _ = Fast_rcnn(images, x, proposals.detach(), labels, bboxs)
    break
