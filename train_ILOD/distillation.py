import torch, torchvision
import torch.nn as nn
DEVICE = torch.device("cuda:0")
backbone = torchvision.models.vgg16(weights="VGG16_Weights.DEFAULT").features[:30]
backbone.to(DEVICE)


example_tensor1 = torch.randn(3, 224, 224, dtype = torch.float).to(DEVICE)
example_tensor2 = torch.randn(3, 224, 224, dtype = torch.float).to(DEVICE)
output1, output2 = backbone(example_tensor1), backbone(example_tensor2)



def calculate_feature_distillation_loss(source_features, target_features):
    final_feature_distillation_loss = []
    for i in range(len(source_features)):
        source_feature, target_feature = source_features[i],target_features[i]
        source_feature_average, target_feature_average = torch.mean(source_feature), torch.mean(target_feature)
        normalized_source_feature, normalized_target_feature = (source_feature - source_feature_average), (target_feature - target_feature_average)
        feature_difference = normalized_source_feature - normalized_target_feature
        feature_size = feature_difference.size()
        filter = torch.zeros(feature_size).to(DEVICE)
        feature_distillation_loss = torch.max(feature_difference, filter)
        final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
        del filter
        torch.cuda.empty_cache()
    final_feature_distillation_loss = sum(final_feature_distillation_loss)/len(source_features)
    
    return final_feature_distillation_loss

#print(calculate_feature_distillation_loss(output1,output2))

def calculate_rpn_distillation_loss(rpn_objectness_source, rpn_bbox_deltas_source, rpn_objectness_target, rpn_bbox_deltas_target, bbox_threshold):
    objectness_difference = []
    final_rpn_cls_distillation_loss = []
    for i in range(len(rpn_objectness_source)):
        rpn_objectness_difference = rpn_objectness_source[i] - rpn_objectness_target[i]
        objectness_difference.append(rpn_objectness_difference)
        filter = torch.zeros(rpn_objectness_source[i].size()).to(DEVICE)
        rpn_distillation_loss = torch.max(rpn_objectness_difference, filter)
        final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
        del filter
        torch.cuda.empty_cache()
    final_rpn_cls_distillation_loss = sum(final_rpn_cls_distillation_loss)/len(rpn_objectness_source)
    
    final_rpn_bbs_distillation_loss = []
    l2_loss = nn.MSELoss(size_average = False, reduce = False)
    
    for i in range(len(rpn_bbox_deltas_source)):
        current_source_rpn_bbox, current_target_rpn_bbox = rpn_bbox_deltas_source[i], rpn_bbox_deltas_target[i]
        current_objectness_difference = objectness_difference[i]
        current_objectness_mask = current_objectness_difference.clone()
        current_objectness_mask[current_objectness_difference > bbox_threshold] = 1
        current_objectness_mask[current_objectness_difference <= bbox_threshold] = 0
        masked_source_rpn_bbox = current_source_rpn_bbox * current_objectness_mask
        masked_target_rpn_bbox = current_target_rpn_bbox * current_objectness_mask
        
        current_bbox_distillation_loss = l2_loss(masked_source_rpn_bbox, masked_target_rpn_bbox)
        final_rpn_bbs_distillation_loss.append(torch.mean(torch.mean(torch.sum(current_bbox_distillation_loss,dim=2),dim=1),dim=0))
    
    final_rpn_bbs_distillation_loss = sum(final_rpn_bbs_distillation_loss)/len(rpn_bbox_deltas_source)
    
    final_rpn_loss = final_rpn_cls_distillation_loss + final_rpn_bbs_distillation_loss
    
    return final_rpn_loss
        
def calculate_roi_distillation_looses(model_source, model_target, images):
    
    return None