from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import math

from utils.utils import dense_to_one_hot

class Jde_RCNN(GeneralizedRCNN):
    def __init__(self, backbone, num_ID, num_classes=2, len_embeddings=128,
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

        out_channels = backbone.out_channels

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
            box_predictor = JDEPredictor(
                representation_size,
                num_classes,
                len_embeddings
                )

        roi_heads = JDE_RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            len_embeddings, num_ID)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(Jde_RCNN, self).__init__(backbone, rpn, roi_heads, transform)




class JDEPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes, size_embedding):
        super(JDEPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.extract_embedding = nn.Linear(in_channels, size_embedding)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        embedding = self.extract_embedding(x)

        return scores, bbox_deltas, embedding

class JDE_RoIHeads(RoIHeads):
    def __init__(self,
                box_roi_pool,
                box_head,
                box_predictor,
                # Faster R-CNN training
                fg_iou_thresh, bg_iou_thresh,
                batch_size_per_image, positive_fraction,
                bbox_reg_weights,
                # Faster R-CNN inference
                score_thresh,
                nms_thresh,
                detections_per_img,
                # ReID parameters
                len_embeddings, num_ID,
                # Mask
                mask_roi_pool=None,
                mask_head=None,
                mask_predictor=None,
                keypoint_roi_pool=None,
                keypoint_head=None,
                keypoint_predictor=None,
                ):

        super(JDE_RoIHeads, self).__init__(box_roi_pool, box_head, box_predictor,
                                          fg_iou_thresh, bg_iou_thresh,
                                          batch_size_per_image, positive_fraction,
                                          bbox_reg_weights,
                                          score_thresh, nms_thresh, detections_per_img)
                                          
        self.num_ID = num_ID
        self.len_embeddings = len_embeddings
        self.identifier = nn.Linear(len_embeddings, num_ID)

        # TODO
        self.s_c = nn.Parameter(-4.15*torch.ones(1))  # -4.15
        self.s_r = nn.Parameter(-4.85*torch.ones(1))  # -4.85
        self.s_id = nn.Parameter(-2.3*torch.ones(1))  # -2.3
        self.emb_scale = math.sqrt(2) * math.log(self.num_ID-1) if self.num_ID>1 else 1

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint:
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
            ids = labels.copy()
            labels = [(label>0).long() for label in labels]
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression, embeddings = self.box_predictor(box_features)

        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg, loss_reid = self.JDE_loss(
                class_logits, box_regression, embeddings, labels, regression_targets, ids)
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_reid=loss_reid)
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                        embeddings=embeddings[i]
                    )
                )

        # if self.has_mask:
        #     mask_proposals = [p["boxes"] for p in result]
        #     if self.training:
        #         # during training, only focus on positive boxes
        #         num_images = len(proposals)
        #         mask_proposals = []
        #         pos_matched_idxs = []
        #         for img_id in range(num_images):
        #             pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
        #             mask_proposals.append(proposals[img_id][pos])
        #             pos_matched_idxs.append(matched_idxs[img_id][pos])

        #     mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
        #     mask_features = self.mask_head(mask_features)
        #     mask_logits = self.mask_predictor(mask_features)

        #     loss_mask = {}
        #     if self.training:
        #         gt_masks = [t["masks"] for t in targets]
        #         gt_labels = [t["labels"] for t in targets]
        #         loss_mask = maskrcnn_loss(
        #             mask_logits, mask_proposals,
        #             gt_masks, gt_labels, pos_matched_idxs)
        #         loss_mask = dict(loss_mask=loss_mask)
        #     else:
        #         labels = [r["labels"] for r in result]
        #         masks_probs = maskrcnn_inference(mask_logits, labels)
        #         for mask_prob, r in zip(masks_probs, result):
        #             r["masks"] = mask_prob

        #     losses.update(loss_mask)

        # if self.has_keypoint:
        #     keypoint_proposals = [p["boxes"] for p in result]
        #     if self.training:
        #         # during training, only focus on positive boxes
        #         num_images = len(proposals)
        #         keypoint_proposals = []
        #         pos_matched_idxs = []
        #         for img_id in range(num_images):
        #             pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
        #             keypoint_proposals.append(proposals[img_id][pos])
        #             pos_matched_idxs.append(matched_idxs[img_id][pos])

        #     keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
        #     keypoint_features = self.keypoint_head(keypoint_features)
        #     keypoint_logits = self.keypoint_predictor(keypoint_features)

        #     loss_keypoint = {}
        #     if self.training:
        #         gt_keypoints = [t["keypoints"] for t in targets]
        #         loss_keypoint = keypointrcnn_loss(
        #             keypoint_logits, keypoint_proposals,
        #             gt_keypoints, pos_matched_idxs)
        #         loss_keypoint = dict(loss_keypoint=loss_keypoint)
        #     else:
        #         keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
        #         for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
        #             r["keypoints"] = keypoint_prob
        #             r["keypoints_scores"] = kps

        #     losses.update(loss_keypoint)

        return result, losses
        
    
    def JDE_loss(self, class_logits, box_regression, embeddings, labels, regression_targets, ids):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        ids = torch.cat(ids, dim=0)

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, -1, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        reid_logits = self.identifier(embeddings)
        # ids = dense_to_one_hot(np.array(ids)-1, self.num_ID)
        reid_loss = F.cross_entropy(reid_logits, ids)

        return classification_loss, box_loss, reid_loss

