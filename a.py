import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from model import Jde_RCNN

backbone = resnet_fpn_backbone('resnet50', True)
backbone.out_channels = 256

model = Jde_RCNN(backbone, num_ID=10, min_size=720, max_size=1280, version='v1')

imgs = torch.rand((1,3,1280,720))
boxes = [torch.Tensor([[20,30,40,50]]),torch.Tensor([[20,30,40,50]])]
print(type(boxes))
model.eval()
model.load_image(imgs.clone())
boxes, scores = model.detect()
# result = model.predict_boxes(boxes)
embedding = model.get_embedding(boxes)
print(embedding.size())