import torch
import torch.nn as nn
import torchvision.models as models

class CarlaBinaryClassifier(nn.Module):
    """
    A binary classifier wrapping a ResNet-18 backbone for targeted
    perception tasks within the CARLA autonomous driving environment.
    """
    def __init__(self, backbone_name='resnet18', pretrained=True):
        super(CarlaBinaryClassifier, self).__init__()
        
        if backbone_name == 'resnet18':
            # Initialize ResNet-18 backbone weights pre-trained on ImageNet
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            # Isolate the feature maps by disabling the default dense head
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Backbone architecture '{backbone_name}' is not currently supported.")
            
        # Define the binary classification sequence projecting features to a single logit
        self.classifier_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        """
        Executes a forward pass across the feature extraction and classification sequence.
        Args:
            x (torch.Tensor): Input tensor batch of shape (Batch Size, 3, H, W)
        Returns:
            torch.Tensor: Raw single unscaled logit per batch entry
        """
        features = self.backbone(x)
        logits = self.classifier_head(features)
        return logits

def get_carla_perception_system():
    """
    Instantiates the three separate, physically decoupled perception models
    mandated by the autonomous safety system requirements specification.
    """
    pedestrian_model = CarlaBinaryClassifier(backbone_name='resnet18', pretrained=True)
    vehicle_model = CarlaBinaryClassifier(backbone_name='resnet18', pretrained=True)
    traffic_light_model = CarlaBinaryClassifier(backbone_name='resnet18', pretrained=True)
    
    return pedestrian_model, vehicle_model, traffic_light_model