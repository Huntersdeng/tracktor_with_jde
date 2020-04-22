import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_boxes
from torchvision.ops import boxes as box_ops
from model import JDE_RoIHeads, featureExtractor, featureHead
import math

class Jde_RCNN(GeneralizedRCNN):
    def __init__(self, backbone, num_classes=2, num_ID=128,version='v1',
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels
        emb_scale = math.sqrt(2) * math.log(num_ID-1) if num_ID>1 else 1
        len_embeddings = 128

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        representation_size = 1024
        embed_extractor = featureExtractor(
            representation_size,
            len_embeddings,
            emb_scale
            )

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        embed_head = featureHead(
            out_channels * resolution ** 2,
            representation_size)

        roi_heads = JDE_RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            len_embeddings, num_ID, embed_head, embed_extractor)
        roi_heads.version = 'v1'
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(Jde_RCNN, self).__init__(backbone, rpn, roi_heads, transform)
        self.version = version
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None
        self.box_features = None

    def detect(self):
        device = list(self.parameters())[0].device
        images = self.preprocessed_images
        images = images.to(device)
        proposals, _ = self.rpn(images, self.features, None)
        detections, _ = self.roi_heads(self.features, proposals, images.image_sizes, None)
        detections = self.transform.postprocess(detections, images.image_sizes, self.original_image_sizes)[0]
        # if self.version=='v2':
        #     boxes, scores, box_features = detections['boxes'].detach(), detections['scores'].detach(), detections['box_features'].detach()
        #     for box, box_feature in zip(boxes, box_features):
        #         self.box_features[str(int(box[0]))+','+str(int(box[1]))+','+str(int(box[2]))+','+str(int(box[3]))] = box_feature
        # else:
        boxes, scores = detections['boxes'].detach(), detections['scores'].detach()
        return boxes, scores

    def predict_boxes(self, boxes):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]

        box_features = self.roi_heads.box_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        # score_thresh = self.roi_heads.score_thresh
        # nms_thresh = self.roi_heads.nms_thresh

        # self.roi_heads.score_thresh = self.roi_heads.nms_thresh = 1.0
        # self.roi_heads.score_thresh = 0.0
        # self.roi_heads.nms_thresh = 1.0
        # detections, detector_losses = self.roi_heads(
        #     features, [boxes.squeeze(dim=0)], images.image_sizes, targets)

        # self.roi_heads.score_thresh = score_thresh
        # self.roi_heads.nms_thresh = nms_thresh

        # detections = self.transform.postprocess(
        #     detections, images.image_sizes, original_image_sizes)

        # detections = detections[0]
        # return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        pred_boxes = box_ops.clip_boxes_to_image(pred_boxes, self.original_image_sizes[0])
        # if self.version=='v2':
        #     for box, box_feature in zip(pred_boxes, box_features):
        #         self.box_features[str(int(box[0]))+','+str(int(box[1]))+','+str(int(box[2]))+','+str(int(box[3]))] = box_feature
        return pred_boxes, pred_scores

    def get_embedding(self, boxes):
        # if self.version=='v2':
        #     embed_features = reduce(lambda x,y: torch.cat((x,y)), [self.box_features[str(int(box[0]))+','+str(int(box[1]))+','+str(int(box[2]))+','+str(int(box[3]))].view(1,-1) for box in boxes])
        # if self.version=='v1':
        if type(boxes)!=list:
            boxes = [boxes]
        features = self.roi_heads.box_roi_pool(self.features, boxes, self.preprocessed_images.image_sizes)
        embed_features = self.roi_heads.embed_head(features)
        embeddings = self.roi_heads.embed_extractor(embed_features)
        return embeddings

    def load_image(self, images):
        device = list(self.parameters())[0].device
        images = images.to(device)

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.transform(images, None)
        self.preprocessed_images = preprocessed_images

        self.features = self.backbone(preprocessed_images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])
        self.box_features={}
