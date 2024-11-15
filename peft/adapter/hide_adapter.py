import torch.nn as nn
import torch
import numpy as np
import math

class HideAdapter(nn.Module):
    def __init__(self, pool_size, depth, dim, rank, scale=1.0, adapter_layernorm=False, init_option='lora'):
        super(HideAdapter, self).__init__()
        self.pool_size = pool_size
        self.dim = dim
        self.rank = rank
        self.depth = depth
        self.down_proj = nn.Parameter(torch.zeros(self.pool_size, self.depth, self.dim, self.rank))
        self.down_bias = nn.Parameter(torch.zeros(self.pool_size, self.depth, self.rank))
        self.up_proj = nn.Parameter(torch.zeros(self.pool_size, self.depth, self.rank, self.dim))
        self.up_bias = nn.Parameter(torch.zeros(self.pool_size, self.depth, self.dim))
        self.scale = scale
        self.init_option = init_option
        self.adapter_layernorm = adapter_layernorm
        if self.adapter_layernorm:
            self.adapters_layer_norm_weight = nn.Parameter(torch.ones(self.pool_size, self.depth, self.dim))
            self.adapters_layer_norm_bias = nn.Parameter(torch.zeros(self.pool_size, self.depth, self.dim))
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for i in range(self.pool_size):
                for j in range(self.depth):
                    nn.init.kaiming_normal_(self.down_proj[i, j, :, :], a=math.sqrt(5))
                    nn.init.zeros_(self.up_proj[i, j, :, :])
                    nn.init.zeros_(self.down_bias[i, j, :])
                    nn.init.zeros_(self.up_bias[i, j, :])
                    if self.adapter_layernorm:
                        nn.init.ones_(self.adapters_layer_norm_weight[i, j, :])
                        nn.init.zeros_(self.adapters_layer_norm_bias[i, j, :])

    def forward(self, x, task_id=-1, depth_id=-1, add_residual=True, train=False, **kwargs):
        residual = x
        if train:
            assert isinstance(task_id, int)
            down = x @ self.down_proj[task_id, depth_id] + self.down_bias[task_id, depth_id]
            down = self.relu(down)
            up = down @ self.up_proj[task_id, depth_id] + self.up_bias[task_id, depth_id]
            up = up * self.scale
        else:
            down = torch.bmm(x, self.down_proj[task_id, depth_id]) + self.down_bias[task_id, depth_id].unsqueeze(1)
            down = self.relu(down)
            up = torch.bmm(down, self.up_proj[task_id, depth_id]) + self.up_bias[task_id, depth_id].unsqueeze(1)
            up = up * self.scale
        
        if self.adapter_layernorm:
            mean = torch.mean(up, dim=-1, keepdim=True)
            var = torch.var(up, dim=-1, keepdim=True, unbiased=False)
            up = (up - mean) / (var + 1e-5)
            if train:
                up = up * self.adapters_layer_norm_weight[task_id, depth_id] + self.adapters_layer_norm_bias[task_id, depth_id]
            else:
                up = up * self.adapters_layer_norm_weight[task_id, depth_id].unsqueeze(-1) + self.adapters_layer_norm_bias[task_id, depth_id].unsqueeze(-1)

        if add_residual:
            output = up + residual
        else:
            output = up
        
        return output
    
    


