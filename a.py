import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from model import Jde_RCNN

backbone = resnet_fpn_backbone('resnet50', True)
backbone.out_channels = 256

model = Jde_RCNN(backbone, num_ID=10, min_size=300, max_size=400, version='v2')

imgs = torch.rand((1,3,300,400))
boxes = torch.Tensor([[20,30,40,50]])
print(type(boxes))
# model.eval()
# model.load_image(imgs.clone())
# boxes, scores = model.detect()
# # result = model.predict_boxes(boxes)
# embedding = model.get_embedding(boxes)