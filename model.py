from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_boxes
from torchvision.ops import boxes as box_ops

import math
from functools import reduce

class Jde_RCNN(GeneralizedRCNN):
    def __init__(self, backbone, num_ID, num_classes=2, version='v1',
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
                 box_batch_size_per_image=256, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # Embedding parameters
                 len_embeddings=128, embed_head=None, embed_extractor=None):
        
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
                output_size=11,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        emb_scale = math.sqrt(2) * math.log(num_ID-1) if num_ID>1 else 1

        if embed_head is None:
            if version=='v1':
                resolution = box_roi_pool.output_size[0]
                representation_size = 1024
                embed_head = featureHead(
                    out_channels * resolution ** 2,
                    representation_size)
            if version=='v2':
                embed_head = None

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)
        
        if embed_extractor is None:
            representation_size = 1024
            embed_extractor = featureExtractor(
                representation_size,
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
            len_embeddings, num_ID, embed_head, embed_extractor)
        roi_heads.version = version

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
        if self.version=='v2':
            boxes, scores, box_features = detections['boxes'].detach(), detections['scores'].detach(), detections['box_features'].detach()
            for box, box_feature in zip(boxes, box_features):
                self.box_features[str(int(box[0]))+','+str(int(box[1]))+','+str(int(box[2]))+','+str(int(box[3]))] = box_feature
        else:
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
        if self.version=='v2':
            for box, box_feature in zip(pred_boxes, box_features):
                self.box_features[str(int(box[0]))+','+str(int(box[1]))+','+str(int(box[2]))+','+str(int(box[3]))] = box_feature
        return pred_boxes, pred_scores

    def get_embedding(self, boxes):
        if self.version=='v2':
            embed_features = reduce(lambda x,y: torch.cat((x,y)), [self.box_features[str(int(box[0]))+','+str(int(box[1]))+','+str(int(box[2]))+','+str(int(box[3]))].view(1,-1) for box in boxes])
        if self.version=='v1':
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


class featureHead(nn.Module):
    """
    heads for getting embeddings

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(featureHead, self).__init__()

        self.fc8 = nn.Linear(in_channels, representation_size)
        self.fc9 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(F.dropout(self.fc8(x)),0.5)
        x = F.relu(F.dropout(self.fc9(x)),0.5)
        return x

class featureExtractor(nn.Module):
    """
    enbedding extracting layers for our model

    Arguments:
        in_channels    (int): number of input channels
        size_embedding (int): number of the embeddings' dimension
        emb_scale      (int): the scale of embedding
    """

    def __init__(self, in_channels, size_embedding, emb_scale):
        super(featureExtractor, self).__init__()
        self.extract_embedding = nn.Linear(in_channels, size_embedding)
        self.emb_scale = emb_scale

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        embedding = self.extract_embedding(x)
        embedding = self.emb_scale * F.normalize(embedding)

        return embedding

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
                len_embeddings, num_ID, embed_head, embed_extractor,
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
        self.embed_head = embed_head
        self.embed_extractor = embed_extractor
        self.num_ID = num_ID
        self.len_embeddings = len_embeddings
        self.identifier = nn.Linear(len_embeddings, num_ID)

        # # TODO
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
            if self.version == 'v1':
                embed_features = self.embed_head(features)
            elif self.version =='v2':
                embed_features = box_features
            else:
                raise ValueError
            embeddings = self.embed_extractor(embed_features)

        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg, loss_reid = self.JDE_loss(
                class_logits, box_regression, embeddings, labels, regression_targets, ids)

            loss_total = torch.exp(-self.s_r)*loss_box_reg + torch.exp(-self.s_c)*loss_classifier + torch.exp(-self.s_id)*loss_reid + \
                   (self.s_r + self.s_c + self.s_id)
            losses = dict(loss_total=loss_total, loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_reid=loss_reid)
        else:
            if self.version == 'v1':
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
            elif self.version =='v2':
                boxes, scores, labels, box_features = self.postprocess_detections_jde(class_logits, box_regression, box_features, proposals, image_shapes)
                num_images = len(boxes)
                for i in range(num_images):
                    result.append(
                        dict(
                            boxes=boxes[i],
                            labels=labels[i],
                            scores=scores[i],
                            box_features=box_features[i]
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

    
    def postprocess_detections_jde(self, class_logits, box_regression, box_features, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_features = box_features.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_box_features = [] 
        for boxes, scores, features, image_shape in zip(pred_boxes, pred_scores, pred_features, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels, features = boxes[keep], scores[keep], labels[keep], features[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_box_features.append(features)

        return all_boxes, all_scores, all_labels, all_box_features
