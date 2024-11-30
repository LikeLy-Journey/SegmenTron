import torch
import torch.nn as nn
import torch.nn.functional as F
from .segbase import SegBaseModel
from ..modules import BasePixelDecoder, StandardTransformerDecoder
from .model_zoo import MODEL_REGISTRY
from ..config import cfg

__all__ = ['MaskFormer']


@MODEL_REGISTRY.register()
class MaskFormer(SegBaseModel):
    r"""Implements the MaskFormer head.

    See `Per-Pixel Classification is Not All You Need for Semantic
    Segmentation <https://arxiv.org/pdf/2107.06278>`_ for details.
    """

    def __init__(self):
        super(MaskFormer, self).__init__()
        self.in_channels = cfg.MODEL.MASKFORMER.IN_CHANNELS
        self.feat_channels = cfg.MODEL.MASKFORMER.FEAT_CHANNELS
        self.out_channels = cfg.MODEL.MASKFORMER.OUT_CHANNELS
        self.queries = cfg.MODEL.MASKFORMER.QUERIES
        self.decode_layers = cfg.MODEL.MASKFORMER.DECODE_LAYERS
        self.deep_supervision = cfg.MODEL.MASKFORMER.DEEP_SUPERVISION
        self.head = _MaskFormerHead(num_classes=self.nclass,
                                    in_channels=self.in_channels,
                                    feat_channels=self.feat_channels,
                                    out_channels=self.out_channels,
                                    decode_layers=self.decode_layers,
                                    queries=self.queries)

        self.__setattr__('decoder', ['head'])

    def forward(self, x):
        size = x.size()[2:]
        feats = self.encoder(x)
        outputs = self.head(feats)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        # upsample masks
        mask_pred_results = F.interpolate(mask_pred_results, size=size, mode='bilinear', align_corners=False)

        # semantic inference
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_results = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)

        return {"inference_results": seg_results, "loss_results": outputs}


class _MaskFormerHead(nn.Module):
    def __init__(self,
                 num_classes,
                 transformer_in_feature="base",
                 in_channels=(96, 192, 384, 768),
                 feat_channels=256,
                 out_channels=256,
                 decode_layers=6,
                 queries=100,
                 deep_supervision=True):
        super(_MaskFormerHead, self).__init__()
        self.pixel_decoder = BasePixelDecoder(in_channels, feat_channels, out_channels)
        self.transformer_in_feature = transformer_in_feature
        self.predictor = StandardTransformerDecoder(
            in_channels=feat_channels if transformer_in_feature == "transformer_encoder" else in_channels[-1],
            mask_classification=True,
            num_classes=num_classes,
            hidden_dim=256,
            num_queries=queries,
            nheads=8,
            dropout=0.1,
            dim_feedforward=2048,
            enc_layers=0,
            dec_layers=decode_layers,
            pre_norm=False,
            deep_supervision=deep_supervision,
            mask_dim=out_channels,
            enforce_input_project=False)

    def forward(self, x):
        """Forward function.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            predictions (Tensor): The output of the MaskFormer head.
        """
        memory = x[-1]
        mask_features, transformer_encoder_features = self.pixel_decoder(x)
        if self.transformer_in_feature == "transformer_encoder":
            assert (
                    transformer_encoder_features is not None
            ), "Please use the TransformerEncoderPixelDecoder."
            predictions = self.predictor(transformer_encoder_features, mask_features)
        else:
            predictions = self.predictor(memory, mask_features)
        return predictions
