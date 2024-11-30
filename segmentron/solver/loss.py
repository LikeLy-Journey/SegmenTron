"""Custom losses."""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .lovasz_losses import lovasz_softmax
from .matcher import HungarianMatcher
from ..models.pointrend import point_sample
from ..data.dataloader import datasets
from ..utils.distributed import get_world_size
from ..utils.misc import nested_tensor_from_tensor_list, is_dist_avail_and_initialized
from ..config import cfg

__all__ = ['get_segmentation_loss']


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def _multiple_forward(self, *inputs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            loss += super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        elif len(preds) > 1:
            return dict(loss=self._multiple_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))


class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""

    def __init__(self, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, W, H] -> [batch, 1, W, H]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            # prob = prob.masked_fill_(1 - valid_mask, 1)
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        # target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w)

        return self.criterion(pred, target)


class EncNetLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = cfg.MODEL.ENCNET.SE_LOSS
        self.se_weight = cfg.MODEL.ENCNET.SE_WEIGHT
        self.nclass = datasets[cfg.DATASET.NAME].NUM_CLASS
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            return super(EncNetLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(EncNetLoss, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.aux_weight * loss2 + self.se_weight * loss3)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect


class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs))


class LovaszSoftmax(MixSoftmaxCrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(LovaszSoftmax, self).__init__()
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = lovasz_softmax(F.softmax(preds[0], dim=1), target, ignore=self.ignore_index)
        for i in range(1, len(preds)):
            aux_loss = lovasz_softmax(F.softmax(preds[i], dim=1), target, ignore=self.ignore_index)
            loss += self.aux_weight * aux_loss
        return loss

    def _multiple_forward(self, *inputs):
        *preds, target = tuple(inputs)
        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            loss += super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        elif len(preds) > 1:
            return dict(loss=self._multiple_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, aux=True, aux_weight=0.2, ignore_index=-1,
                 size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.aux_weight = aux_weight
        self.size_average = size_average
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = self._base_forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = self._base_forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def _base_forward(self, output, target):

        if output.dim() > 2:
            output = output.contiguous().view(output.size(0), output.size(1), -1)
            output = output.transpose(1, 2)
            output = output.contiguous().view(-1, output.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        logpt = self.ce_fn(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * self.alpha * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        return dict(loss=self._aux_forward(*inputs))


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target, valid_mask):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum((predict.pow(self.p) + target.pow(self.p)) * valid_mask, dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input"""

    def __init__(self, weight=None, aux=True, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.aux = aux
        self.aux_weight = aux_weight

    def _base_forward(self, predict, target, valid_mask):

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[-1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[..., i], valid_mask)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[-1]

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)
        # valid_mask: used to exclude specific elements from the loss calculation (e.g., ignoring background pixels)
        valid_mask = (target != self.ignore_index).long()
        target_one_hot = F.one_hot(torch.clamp_min(target, 0))
        loss = self._base_forward(preds[0], target_one_hot, valid_mask)
        for i in range(1, len(preds)):
            aux_loss = self._base_forward(preds[i], target_one_hot, valid_mask)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        return dict(loss=self._aux_forward(*inputs))


class PointRendLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(PointRendLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index

    def forward(self, *inputs, **kwargs):
        result, gt = tuple(inputs)

        pred = F.interpolate(result["coarse"], gt.shape[-2:], mode="bilinear", align_corners=True)
        seg_loss = F.cross_entropy(pred, gt, ignore_index=self.ignore_index)

        gt_points = point_sample(
            gt.float().unsqueeze(1),
            result["points"],
            mode="nearest",
            align_corners=False
        ).squeeze_(1).long()
        points_loss = F.cross_entropy(result["rend"], gt_points, ignore_index=self.ignore_index)

        loss = seg_loss + points_loss

        return dict(loss=loss)


def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


class MaskFormerLoss(nn.Module):
    """This class computes the loss for DETR.
        The process happens in two steps:
            1) we compute hungarian assignment between ground truth boxes and the outputs of the model
            2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MaskFormerLoss, self).__init__()
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.num_classes = datasets[cfg.DATASET.NAME].NUM_CLASS
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASKFORMER.DEEP_SUPERVISION
        self.eos_coef = cfg.MODEL.MASKFORMER.NO_OBJECT_WEIGHT
        class_weight = cfg.MODEL.MASKFORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASKFORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASKFORMER.MASK_WEIGHT

        self.matcher = HungarianMatcher(cost_class=class_weight, cost_mask=mask_weight, cost_dice=dice_weight)

        self.weight_dict = {"loss_ce": 1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASKFORMER.DECODE_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in self.weight_dict.items()})
            self.weight_dict.update(aux_weight_dict)

        self.losses = ["labels", "masks"]

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(
            src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks, target_masks, num_masks),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def transform_targets(self, targets):
        """
        Transform targets into the format expected by the loss function

        Args:
            targets: Batched tensors with shape [batch_size, H, W], each pixel element is the class label,
                        where 0 <= targets[b, i, j] <= C-1, and C is the number of classes.

        Returns:
            targets: A list of targets (len(targets) = batch_size), where each target is a list containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks
        """
        batch_size = targets.shape[0]
        transformed_targets = []

        for batch in range(batch_size):
            target = targets[batch]
            classes = torch.unique(target, sorted=False, return_inverse=False, return_counts=False)
            gt_labels = classes[classes != self.ignore_index]
            masks = []
            for class_id in gt_labels:
                masks.append(target == class_id)
            if len(masks) == 0:
                gt_masks = torch.zeros((0, target.shape[-2], target.shape[-1])).to(target)
            else:
                gt_masks = torch.stack(masks).squeeze(1)
            transformed_targets.append({
                "labels": gt_labels,
                "masks": gt_masks,
            })

        return transformed_targets

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        targets = self.transform_targets(targets)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        for k in list(losses.keys()):
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses


def get_segmentation_loss(model, use_ohem=False, **kwargs):
    if use_ohem:
        return MixSoftmaxCrossEntropyOHEMLoss(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'lovasz':
        logging.info('Use lovasz loss!')
        return LovaszSoftmax(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'focal':
        logging.info('Use focal loss!')
        return FocalLoss(**kwargs)
    elif cfg.SOLVER.LOSS_NAME == 'dice':
        logging.info('Use dice loss!')
        return DiceLoss(**kwargs)

    model = model.lower()
    if model == 'icnet':
        return ICNetLoss(**kwargs)
    elif model == 'encnet':
        return EncNetLoss(**kwargs)
    elif model == 'pointrend':
        logging.info('Use pointrend loss!')
        return PointRendLoss(**kwargs)
    elif model == 'maskformer':
        logging.info('Use maskformer loss!')
        return MaskFormerLoss(**kwargs)
    else:
        return MixSoftmaxCrossEntropyLoss(**kwargs)
