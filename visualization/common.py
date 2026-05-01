import math
import torch
import torch.nn as nn
from PAT.distillers.pat import PAT, NUM_STAGES, RegionAttention
from PAT.distillers._base import BaseDistiller

# This is just so we can run it on cpu if desired
class DeviceAwarePAT(PAT):
    def __init__(self, device, student, teacher, criterion, input_size, args, **kwargs):
        BaseDistiller.__init__(self, student, teacher, criterion, args) 

        self.student.to(device)
        self.teacher.to(device)

        # the rest is a copy of the original class's init method, but with things unnecessary for the visualization removed 
        feat_s, feat_t = None, None
        with torch.no_grad():
            c, h, w = input_size
            x = torch.rand(1, c, h, w).to(device)
            _, feat_s = self.student(x, requires_feat=True)

        student_shapes = []
        teacher_shapes = []
        for stage in range(1, NUM_STAGES + 1):
            idx_s, _ = self.student.stage_info(stage)
            feat_s_shape = feat_s[idx_s].shape
            if len(feat_s_shape) == 3 and not math.sqrt(feat_s_shape[1]).is_integer():
                b, l, d = feat_s_shape
                feat_s_shape = torch.Size((b, l - 1, d))
            student_shapes.append(feat_s_shape)

        ### construct region-aware distillation module ###
        num_queries = self.args.pat_raa_num_queries  # num_queries = num_stages(4) * num_queries_per_stage
        dim = self.args.pat_raa_dim  # dim = 512
        self.attention_blending = RegionAttention(
            student_shapes, teacher_shapes, num_queries=num_queries, dim=dim, heads=8, dropout=0.0
        )

        self.to(device)
        

