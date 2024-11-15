import torch.nn as nn
import torch
import numpy as np
import math
import copy

class MomentumAdapter(nn.Module):
    def __init__(self, depth, dim, rank, scale=1.0, adapter_layernorm=False, init_option='lora', momentum=0.99):
        super(MomentumAdapter, self).__init__()
        self.dim = dim
        self.rank = rank
        self.depth = depth
        self.down_proj = nn.Parameter(torch.zeros(self.depth, self.dim, self.rank))
        self.down_bias = nn.Parameter(torch.zeros(self.depth, self.rank))
        self.up_proj = nn.Parameter(torch.zeros(self.depth, self.rank, self.dim))
        self.up_bias = nn.Parameter(torch.zeros(self.depth, self.dim))
        self.scale = scale
        self.init_option = init_option
        self.adapter_layernorm = adapter_layernorm
        self.momentum = momentum
        if self.adapter_layernorm:
            self.adapters_layer_norm_weight = nn.Parameter(torch.ones(self.depth, self.dim))
            self.adapters_layer_norm_bias = nn.Parameter(torch.zeros(self.depth, self.dim))
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self): 
        with torch.no_grad():
            for j in range(self.depth):
                nn.init.kaiming_normal_(self.down_proj[j, :, :], a=math.sqrt(5))
                nn.init.zeros_(self.up_proj[j, :, :])
                nn.init.zeros_(self.down_bias[j, :])
                nn.init.zeros_(self.up_bias[j, :])
                if self.adapter_layernorm:
                    nn.init.ones_(self.adapters_layer_norm_weight[j, :])
                    nn.init.zeros_(self.adapters_layer_norm_bias[j, :])

    def forward(self, x, task_id=-1, depth_id=-1, add_residual=True, train=False, old=False):
        residual = x
        if not old:
            down = x @ self.down_proj[depth_id] + self.down_bias[depth_id]
            down = self.relu(down)
            up = down @ self.up_proj[depth_id] + self.up_bias[depth_id]
            up = up * self.scale
            
            if self.adapter_layernorm:
                mean = torch.mean(up, dim=-1, keepdim=True)
                var = torch.var(up, dim=-1, keepdim=True, unbiased=False)
                up = (up - mean) / (var + 1e-5)
                if train:
                    up = up * self.adapters_layer_norm_weight[depth_id] + self.adapters_layer_norm_bias[depth_id]
                else:
                    up = up * self.adapters_layer_norm_weight[depth_id].unsqueeze(-1) + self.adapters_layer_norm_bias[depth_id].unsqueeze(-1)
        else:
            down = x @ self.down_proj_mom[depth_id] + self.down_bias_mom[depth_id]
            down = self.relu(down)
            up = down @ self.up_proj_mom[depth_id] + self.up_bias_mom[depth_id]
            up = up * self.scale
            
            if self.adapter_layernorm:
                mean = torch.mean(up, dim=-1, keepdim=True)
                var = torch.var(up, dim=-1, keepdim=True, unbiased=False)
                up = (up - mean) / (var + 1e-5)
                if train:
                    up = up * self.adapters_layer_norm_weight_mom[depth_id] + self.adapters_layer_norm_bias_mom[depth_id]
                else:
                    up = up * self.adapters_layer_norm_weight_mom[depth_id].unsqueeze(-1) + self.adapters_layer_norm_bias_mom[depth_id].unsqueeze(-1)

        if add_residual:
            output = up + residual
        else:
            output = up
        
        return output
    
    def after_task(self, task_id=-1, device=None):
        if task_id == 0:
            self.down_proj_mom = copy.deepcopy(self.down_proj.detach().clone()).to(device)
            self.down_bias_mom = copy.deepcopy(self.down_bias_mom.detach().clone()).to(device)
            self.up_proj_mom = copy.deepcopy(self.up_proj.detach().clone()).to(device)
            self.up_bias_mom = copy.deepcopy(self.up_bias.detach().clone()).to(device)
            if self.adapter_layernorm:
                self.adapters_layer_norm_weight_mom = copy.deepcopy(self.adapters_layer_norm_weight.detach().clone()).to(device)
                self.adapters_layer_norm_bias_mom = copy.deepcopy(self.adapters_layer_norm_bias.detach().clone()).to(device)
        else:
            self.down_proj_mom = self.momentum * self.down_proj_mom + (1 - self.momentum) * copy.deepcopy(self.down_proj.detach().clone()).to(device)
            self.down_bias_mom = self.momentum * self.down_bias_mom + (1 - self.momentum) * copy.deepcopy(self.down_bias_mom.detach().clone()).to(device)
            self.up_proj_mom = self.momentum * self.up_proj_mom + (1 - self.momentum) * copy.deepcopy(self.up_proj.detach().clone()).to(device)
            self.up_bias_mom = self.momentum * self.up_bias_mom + (1 - self.momentum) * copy.deepcopy(self.up_bias.detach().clone()).to(device)
            if self.adapter_layernorm:
                self.adapters_layer_norm_weight_mom = self.momentum * self.adapters_layer_norm_weight + (1 - self.momentum) * copy.deepcopy(self.adapters_layer_norm_weight.detach().clone()).to(device)
                self.adapters_layer_norm_bias_mom = self.momentum * self.adapters_layer_norm_bias + (1 - self.momentum) * copy.deepcopy(self.adapters_layer_norm_bias.detach().clone()).to(device)

            
    
