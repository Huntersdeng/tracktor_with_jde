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
                 min_size=480, max_size=640,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.5, rpn_bg_iou_thresh=0.4,
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
            anchor_sizes = ((16,22), (32,45), (64,90), (128,181), (256,362))
            aspect_ratios = ((1/3,),) * len(anchor_sizes)
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

        emb_scale = math.sqrt(2) * math.log(num_ID-1) if num_ID>1 else 1

        if box_predictor is None:
            representation_size = 1024
            box_predictor = JDEPredictor(
                representation_size,
                num_classes,
                len_embeddings,
                emb_scale
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
    Standard classification + bounding box regression + enbedding extracting layers
    for our model

    Arguments:
        in_channels    (int): number of input channels
        num_classes    (int): number of output classes (including background)
        size_embedding (int): number of the embeddings' dimension
        emb_scale      (int): the scale of embedding
    """

    def __init__(self, in_channels, num_classes, size_embedding, emb_scale):
        super(JDEPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.extract_embedding = nn.Linear(in_channels, size_embedding)
        self.emb_scale = emb_scale

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        embedding = self.extract_embedding(x)
        embedding = self.emb_scale * F.normalize(embedding)

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
                assert t["ids"].dtype == torch.int64, 'target ids must of int64 type'
                if self.has_keypoint:
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets, ids = self.select_training_samples(proposals, targets)
        
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression, embeddings = self.box_predictor(box_features)

        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg, loss_reid = self.JDE_loss(
                class_logits, box_regression, embeddings, labels, regression_targets, ids)

            loss_total = torch.exp(-self.s_r)*loss_box_reg + torch.exp(-self.s_c)*loss_classifier \
                 + torch.exp(-self.s_id)*loss_reid + (self.s_r + self.s_c + self.s_id)
            loss_total *= 0.5
            losses = dict(loss_total=loss_total, loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_reid=loss_reid)
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

        # compute the reid loss for positive targets
        reid_logits = self.identifier(embeddings)
        index = ids > -1
        if torch.sum(index):
            reid_loss = F.cross_entropy(reid_logits[index], ids[index])
        else:
            reid_loss = torch.tensor(0)
        return classification_loss, box_loss, reid_loss

    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        gt_boxes = [t["boxes"] for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_ids = [t["ids"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels, ids = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, gt_ids)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            ids[img_id] = ids[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, ids

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, gt_ids):
        matched_idxs = []
        labels = []
        ids = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_ids_in_image in zip(proposals, gt_boxes, gt_labels, gt_ids):
            match_quality_matrix = self.box_similarity(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)

            ids_in_image = gt_ids_in_image[clamped_matched_idxs_in_image]
            ids_in_image = ids_in_image.to(dtype=torch.int64) 

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = 0
            ids_in_image[bg_inds] = -1

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler
            ids_in_image[ignore_inds] = -1

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            ids.append(ids_in_image)
        return matched_idxs, labels, ids
