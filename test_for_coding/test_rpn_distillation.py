import torch
import torch.nn as nn
DEVICE = torch.device('cuda:0')

#input torch.Size([1, 16650]) torch.Size([1, 16650, 4])
example_tensor1 = torch.randn(1, 16650, dtype = torch.float).to(DEVICE)
example_tensor2 = torch.randn(1, 16650, dtype = torch.float).to(DEVICE)
example_tensor3 = torch.randn(1, 16650, 4, dtype = torch.float).to(DEVICE)
example_tensor4 = torch.randn(1, 16650, 4, dtype = torch.float).to(DEVICE)

def calculate_rpn_distillation_loss(rpn_objectness_source, rpn_bbox_deltas_source, rpn_objectness_target, rpn_bbox_deltas_target, bbox_threshold):
    # 추현창 이거 고쳐야댐......
    objectness_difference = []
    final_rpn_cls_distillation_loss = []
    print("length of tpn_objectness_source",len(rpn_objectness_source))
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
    #print("length of rpn_bbox_deltas_source",len(rpn_bbox_deltas_source))
    for i in range(len(rpn_bbox_deltas_source)):
        current_source_rpn_bbox, current_target_rpn_bbox = rpn_bbox_deltas_source[i], rpn_bbox_deltas_target[i]
        current_objectness_difference = objectness_difference[i]
        current_objectness_mask = current_objectness_difference.clone()
        current_objectness_mask[current_objectness_difference > bbox_threshold] = 1
        current_objectness_mask[current_objectness_difference <= bbox_threshold] = 0
        masked_source_rpn_bbox = current_source_rpn_bbox * current_objectness_mask
        masked_target_rpn_bbox = current_target_rpn_bbox * current_objectness_mask
        
        current_bbox_distillation_loss = l2_loss(masked_source_rpn_bbox, masked_target_rpn_bbox)
        current_bbox_distillation_loss = torch.mean(current_bbox_distillation_loss)/4
        #final_rpn_bbs_distillation_loss.append(torch.mean(torch.mean(torch.sum(current_bbox_distillation_loss,dim=2),dim=1),dim=0))
        final_rpn_bbs_distillation_loss.append(current_bbox_distillation_loss)
    final_rpn_bbs_distillation_loss = sum(final_rpn_bbs_distillation_loss)/len(rpn_bbox_deltas_source)
    
    final_rpn_loss = final_rpn_cls_distillation_loss + final_rpn_bbs_distillation_loss
    
    return final_rpn_loss

print(calculate_rpn_distillation_loss(example_tensor1[0], example_tensor3[0], example_tensor2[0], example_tensor4[0], 0.7))