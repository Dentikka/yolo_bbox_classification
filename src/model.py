import torch
from torch import nn, Tensor

import timm


class BboxClassificationModel(nn.Module):
    """
    A class to make a model consisting of an embedding model (backbone)
    and classifier
    Currently maintained architectures are:
        MobileNet, EfficientNet, ConvNext, ResNet, ViT
    """
    def __init__(self,
                 cfg_model: dict, 
                 n_classes: int):
        super().__init__()
        self.emb_model = timm.create_model(cfg_model['model'], pretrained=cfg_model['pretrained'])
        self.emb_size = self.emb_model.num_features
        self.emb_model.reset_classifier(0)  # a simpler way to get emb_model from a timm model

        self.classifier = nn.Sequential(
            nn.Linear(self.emb_size, 20),
            nn.ReLU(),
            nn.Linear(20, n_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        emb = self.emb_model(x)
        return self.classifier(emb)

    def set_backbone_state(self, state: str) -> None:
        """
        to freeze or unfreeze model backbone
        state (str): one of 'freeze', 'unfreeze' 
        """
        assert state in ['freeze', 'unfreeze']
        for param in self.emb_model.parameters():
            if state == 'freeze':
                param.requires_grad = False
            elif state == 'unfreeze':
                param.requires_grad = True
    

def get_model(cfg_model: dict,
              n_classes: int,
              device: torch.device='cpu',
              compile: bool=False) -> BboxClassificationModel:
    model = BboxClassificationModel(cfg_model, n_classes)

    model.to(device)
    if compile:
        model = torch.compile(model, dynamic=True)

    return model
