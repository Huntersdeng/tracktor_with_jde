import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from model import Jde_RCNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weights-path', type=str)
parser.add_argument('--len-embed', type=int, default=128, help='length of embeddings')
    
opt = parser.parse_args()
print(opt, end='\n\n')
weights_path = opt.weights_path
with open(os.path.join(weights_path, 'model.yaml'), 'r') as f:
    cfg = yaml.load(f,Loader=yaml.FullLoader)
weights = os.path.join(weights_path, 'latest.pt')
img_size = (cfg['width'], cfg['height'])
backbone_name = cfg['backbone_name']
# Initialize model
backbone = resnet_fpn_backbone(backbone_name, True)
backbone.out_channels = 256
# model = FRCNN_FPN(num_classes=2)

model = Jde_RCNN(backbone, num_ID=cfg['num_ID'], min_size=img_size[1], max_size=img_size[0],len_embeddings=opt.len_embed)
checkpoint = torch.load(weights, map_location='cpu')
layer = ['roi_heads.embed_head.fc8.weight',
            'roi_heads.embed_head.fc8.bias',
            'roi_heads.embed_head.fc9.weight',
            'roi_heads.embed_head.fc9.bias',
            'roi_heads.embed_extractor.extract_embedding.weight',
            'roi_heads.embed_extractor.extract_embedding.bias',
            'roi_heads.identifier.weight',
            'roi_heads.identifier.bias']
weights = checkpoint['model']
epoch_det = checkpoint['epoch_det']
epoch_reid = 0
for name in layer:
    weights.pop(name)
print(model.load_state_dict(weights, strict=False))

checkpoint_new = {'epoch_det': epoch_det,
                  'epoch_reid': epoch_reid,
                  'model': model.state_dict()
                    }

checkpoint = {'epoch_det': epoch_det,
                  'epoch_reid': epoch_reid,
                  'model': weights
                    }

latest = osp.join(weights_path, 'latest.pt')
torch.save(checkpoint_new, latest)
torch.save(checkpoint, osp.join(weights_path, "weights_epoch_" + str(epoch_det) + '_' + str(epoch_reid) + ".pt"))
