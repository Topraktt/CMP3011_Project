import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        obj_mask = targets[..., 4] == 1
        noobj_mask = targets[..., 4] == 0
        
        pred_boxes = predictions[..., :4]
        pred_conf = predictions[..., 4]
        pred_classes = predictions[..., 5:]
        
        target_boxes = targets[..., :4]
        target_conf = targets[..., 4]
        target_classes = targets[..., 5:]
        
        pred_boxes_obj = pred_boxes[obj_mask]
        target_boxes_obj = target_boxes[obj_mask]
        coord_loss = self.lambda_coord * self.mse_loss(pred_boxes_obj, target_boxes_obj)
        
        pred_conf_obj = pred_conf[obj_mask]
        target_conf_obj = target_conf[obj_mask]
        obj_loss = self.bce_loss(pred_conf_obj, target_conf_obj)
        
        pred_conf_noobj = pred_conf[noobj_mask]
        target_conf_noobj = target_conf[noobj_mask]
        noobj_loss = self.lambda_noobj * self.bce_loss(pred_conf_noobj, target_conf_noobj)
        
        pred_classes_obj = pred_classes[obj_mask]
        target_classes_obj = target_classes[obj_mask].argmax(dim=-1)
        class_loss = self.ce_loss(pred_classes_obj, target_classes_obj)
        
        total_loss = coord_loss + obj_loss + noobj_loss + class_loss
        
        return total_loss
