from .model import Jde_RCNN
import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import yaml
import os


class mymodel(torch.nn.Module):
    def __init__(self, weights_path):
        with open(os.path.join(weights_path, 'model.yaml'), 'r') as f:
            cfg = yaml.load(f,Loader=yaml.FullLoader)
        weights = os.path.join(weights_path, 'latest.pt')
        img_size = (cfg['width'], cfg['height'])
        backbone_name = cfg['backbone_name']
        # Initialize model
        backbone = resnet_fpn_backbone(backbone_name, True)
        backbone.out_channels = 256

        model = Jde_RCNN(backbone, num_ID=cfg['num_ID'], min_size=img_size[1], max_size=img_size[0],len_embeddings=cfg['len_embedding'])
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        model.eval()
        super(mymodel, self).__init__()
        self.model = model
    
    def forward(self, img):
        device = list(self.parameters())[0].device
        

        detections = self.model.forward(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()
