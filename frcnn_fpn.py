from collections import OrderedDict
import math

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.roi_heads import RoIHeads

from model import featureExtractor, featureHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead



class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_ID, num_classes=2, len_embeddings=128):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes, min_size=720, max_size=960)
        # these values are cached to allow for feature reuse
        emb_scale = math.sqrt(2) * math.log(num_ID-1) if num_ID>1 else 1
        out_channels = backbone.out_channels

        box_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=7,
                sampling_ratio=2)

        representation_size = 1024

        resolution = box_roi_pool.output_size[0]
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)

        resolution = box_roi_pool.output_size[0]
        embed_head = featureHead(
            out_channels * resolution ** 2,
            representation_size)  
        
        embed_extractor = featureExtractor(
            representation_size,
            len_embeddings,
            emb_scale
            )

        roi_heads = JDE_RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            len_embeddings, num_ID, embed_head, embed_extractor)
        self.roi_heads = roi_heads
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()

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
        return pred_boxes, pred_scores

    def get_embedding(self, boxes):
        features = self.roi_head.box_roi_pool(self.features, boxes, self.preprocessed_images.image_sizes[0])
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


class JDE_RoIHeads(RoIHeads):
    def __init__(self,
                box_roi_pool,
                box_head,
                box_predictor,
                # ReID parameters
                len_embeddings, num_ID, embed_head, embed_extractor,
                ):
        score_thresh=0.05
        nms_thresh=0.5
        detections_per_img=100
        fg_iou_thresh=0.5
        bg_iou_thresh=0.5
        batch_size_per_image=512
        positive_fraction=0.25
        bbox_reg_weights=None
        super(JDE_RoIHeads, self).__init__(box_roi_pool, box_head, box_predictor,
                                          fg_iou_thresh, bg_iou_thresh,
                                          batch_size_per_image, positive_fraction,
                                          bbox_reg_weights,
                                          score_thresh, nms_thresh, detections_per_img)
        self.embed_head = embed_head
        self.embed_extractor = embed_extractor
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
        
        features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(features)
        class_logits, box_regression= self.box_predictor(box_features)


        if self.training:
            embed_features = self.embed_head(features)
            embeddings = self.embed_extractor(embed_features)

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
                        scores=scores[i]
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