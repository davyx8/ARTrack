"""
Basic OSTrack model.
"""
from copy import deepcopy
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.artrackv2_seq.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.layers.mask_decoder import build_maskdecoder

from lib.models.layers.head import build_decoder, MLP, DropPathAllocator

import time


class ARTrackV2Seq(nn.Module):
    """ This is the base class for ARTrackV2Seq """

    def __init__(self, transformer,
                 cross_2_decoder,
                 score_mlp,
                 ):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.score_mlp = score_mlp

        self.identity = torch.nn.Parameter(torch.zeros(1, 3, 768))
        self.identity = trunc_normal_(self.identity, std=.02)

        self.cross_2_decoder = cross_2_decoder

    def forward(self, template, appearance_features, search, seqs_input):
        # Forward pass through the backbone
        feat_dict = self.backbone.forward_features(template, appearance_features, search, self.identity, seqs_input)

        # Extract features for the decoder
        # Assuming 'feat_dict' contains necessary features for cross_2_decoder
        # You might need to adjust these keys based on the actual output of your backbone
        # For example, if backbone outputs 'pred_boxes' and 'pred_masks'
        # predicted_tokens = feat_dict.get('pred_boxes')
        # sequence_features = feat_dict.get('pred_masks')

        # Placeholder for actual output extraction based on model's internal structure
        # This part needs to be aligned with the actual model's output
        output, z_1_feat, score_feat = feat_dict

        predicted_tokens = output['predicted_tokens']
        sequence_scores = output['sequence_scores']
        sequence_features = output['sequence_features']
        score = self.score_mlp(score_feat) # Corrected: score is derived from score_feat
        refined_appearance_features = z_1_feat

        return predicted_tokens, sequence_scores, sequence_features, score, refined_appearance_features

    def track_onnx(self, ort_session, **kwargs):
        template = kwargs['template']
        appearance_features = kwargs['appearance_features']
        search = kwargs['search']
        seq_input = kwargs['seq_input']
        
        ort_inputs = {
            'template': template,
            'appearance_features': appearance_features,
            'search': search,
            'seq_input': seq_input
        }
        
        ort_outputs = ort_session.run(None, ort_inputs)
        
        out = {
            'predicted_tokens': torch.from_numpy(ort_outputs[0]),
            'sequence_scores': torch.from_numpy(ort_outputs[1]),
            'sequence_features': torch.from_numpy(ort_outputs[2]),
            'score': torch.from_numpy(ort_outputs[3]),
            'refined_appearance_features': torch.from_numpy(ort_outputs[4])
        }
        
        return out

class MlpScoreDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, bn=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        out_dim = 1 # score
        if bn:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])
        else:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Linear(n, k)
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])

    def forward(self, reg_tokens):
        """
        reg tokens shape: (b, 4, embed_dim)
        """
        x = self.layers(reg_tokens) # (b, 4, 1)
        x = x.mean(dim=1)   # (b, 1)
        return x

def build_score_decoder(cfg, hidden_dim):
    return MlpScoreDecoder(
        in_dim=hidden_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        bn=False
    )

def build_artrackv2_seq(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('ARTrackV2' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION, prenum=cfg.MODEL.PRENUM)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        print("i use vit_large")
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION, prenum=cfg.MODEL.PRENUM)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    cross_2_decoder = build_maskdecoder(cfg, hidden_dim)

    drop_path = cfg.MODEL.DROP_PATH
    drop_path_allocator = DropPathAllocator(drop_path)
    num_heads = cfg.MODEL.NUM_HEADS
    mlp_ratio = cfg.MODEL.MLP_RATIO
    qkv_bias = cfg.MODEL.QKV_BIAS
    drop_rate = cfg.MODEL.DROP_RATE
    attn_drop = cfg.MODEL.ATTN_DROP
    score_mlp = build_score_decoder(cfg, hidden_dim)

    model = ARTrackV2Seq(
        backbone,
        cross_2_decoder,
        score_mlp,
    )
    load_from = cfg.MODEL.PRETRAIN_PTH
    checkpoint = torch.load(load_from, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
    if 'sequence' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
