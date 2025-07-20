import torch
import torch.nn as nn
import copy

class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False, 
                 use_prefix_mlp=False, bottleneck_size=800):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.use_prefix_mlp = use_prefix_mlp
        self.embed_dim = embed_dim

        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length, 
                                        embed_dim)

                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    #prompt_pool_shape = (2, self.num_layers, self.pool_size, self.length, embed_dim)
                    prompt_pool_shape = (self.pool_size, 2, self.num_layers, self.length, embed_dim)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, pool_size, length, embed_dim
                        nn.init.uniform_(self.prompt, -1, 1)
            else:
                prompt_pool_shape = (self.num_layers, self.pool_size, self.length, embed_dim)  # TODO fix self.num_layers = 1
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)

        if self.use_prefix_mlp:
            self.prefix_MLP_key_layer = nn.ModuleList([nn.Sequential(
            nn.Linear(768, bottleneck_size),
            nn.ReLU(),
            nn.Linear(bottleneck_size, 768),
            ) for _ in range(self.pool_size)])
            self.prefix_MLP_value_layer = nn.ModuleList([nn.Sequential(
            nn.Linear(768, bottleneck_size),
            nn.ReLU(),
            nn.Linear(bottleneck_size, 768),
            ) for _ in range(self.pool_size)])
            # self.prefix_MLP_key_layer = nn.ModuleList([
            # nn.Linear(768, 768)
            # for i in range(self.num_layers)]) 
            # self.prefix_MLP_value_layer = nn.ModuleList([
            # nn.Linear(768, 768)
            # for i in range(self.num_layers)]) 
            self.prompt_copy = torch.zeros(prompt_pool_shape)

            
    
    def forward(self, x_embed, train=False, prompt_mask=None, prompt_idx=None, task_id=-1, **kwargs):
        # assert prompt_mask is not None or prompt_idx is not None or prompt_weight is not None
        assert self.prompt_pool, "In HiDe-Prompt, 'prompt_pool' must be set to True"
        out = dict()
        if self.prompt_pool:
            idx = prompt_idx

            if self.batchwise_prompt and prompt_idx is not None:
                prompt_id, id_counts = torch.unique(prompt_idx, return_counts=True, sorted=True)
                
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(prompt_idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                major_prompt_id = prompt_id[major_idx] # top_k
                # expand to batch
                idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous()  # B, top_k
            
            if prompt_mask is not None:
                idx = prompt_mask  # B, top_k
            if idx is not None:
                out['prompt_idx'] = idx
            if self.use_prefix_tune_for_e_prompt:
                # if prompt_weight is not None:
                #     batched_prompt_raw = torch.einsum("bp,ndplhe->ndblhe", prompt_weight, self.prompt) # num_layers, 2, B, top_k, length, C
                #     # batched_prompt_raw = batched_prompt_raw.unsqueeze(3)
                    # num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                    # # print(top_k)
                    # batched_prompt = batched_prompt_raw.reshape(
                    #     num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                    # )
                # elif prompt_momentum > 0 and prompt_mask is not None:
                #     with torch.no_grad():
                #         batched_prompt_momentum = self.prompt[:, :, 0:idx[0][0]].detach().clone().mean(2, keepdim=True).unsqueeze(2).repeat(1,1,idx.shape[0],1,1,1,1)
                #     batched_prompt_raw = (1-prompt_momentum) * self.prompt[:, :, idx] + prompt_momentum * batched_prompt_momentum
                #     # num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                #     # batched_prompt = batched_prompt_raw.reshape(
                #     #     num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                #     # )
                #else:
                if train: 
                    if self.use_prefix_mlp:
                        batched_prompt_i_0 = (self.prompt[idx, 0] + self.prefix_MLP_key_layer[task_id](self.prompt[idx, 0]))
                        batched_prompt_i_1 = (self.prompt[idx, 1] + self.prefix_MLP_value_layer[task_id](self.prompt[idx, 1]))
                        batched_prompt_raw = torch.stack([batched_prompt_i_0, batched_prompt_i_1], dim=2)
                        #atched_prompt_raw = torch.stack(batched_prompt_raw, dim=0)
                        #batched_prompt_raw = batched_prompt_raw[:, :, idx]
                        print(batched_prompt_raw.shape)
                    
                    else:
                        batched_prompt_raw = self.prompt[idx].squeeze(1)  # B, top_k, dual, num_layers, length, C
                else:
                    if self.use_prefix_mlp:
                        batched_prompt_raw = self.prompt_copy[idx]
                    else:
                        batched_prompt_raw = self.prompt[idx]
                batched_prompt_raw = batched_prompt_raw.unsqueeze(1) 
                batch_size, top_k, dual, num_layers, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    batch_size, top_k, dual, num_layers, length, self.num_heads, embed_dim // self.num_heads
                )
                batch_size, top_k, dual, num_layers, length, num_heads, heads_embed_dim = batched_prompt.shape
                # batched_prompt = batched_prompt.permute(1, 2, 0, 3, 4, 5, 6).contiguous()
                assert top_k == 1, "top_k must be 1 for prefix tuning"
                #batched_prompt = batched_prompt.view(
                #    num_layers, batch_size, dual, top_k*length, num_heads, heads_embed_dim
                #)
                # return shape: batch_size, dual, num_layers, length, num_heads, heads_embed_dim
                batched_prompt = batched_prompt.squeeze(1) 
                
            else:
                # if prompt_weight is not None:
                #     batched_prompt_raw = torch.einsum("bp,npld->nbpld", prompt_weight, self.prompt)
                #     num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                #     batched_prompt = batched_prompt_raw.reshape(
                #         num_layers, batch_size, top_k * length, embed_dim
                #     )
                # else:
                batched_prompt_raw = self.prompt[:, idx]
                num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, top_k * length, embed_dim
                )
        
        out['batched_prompt'] = batched_prompt

        return out
    
    def after_task(self, task_id=-1, device=None, **kwargs):
        if self.use_prefix_mlp:
            with torch.no_grad():
                self.prompt_copy[task_id, 0] = self.prompt[task_id, 0].detach().clone() + self.prefix_MLP_key_layer[task_id](self.prompt[task_id, 0]).detach().clone()
                self.prompt_copy[task_id, 1] = self.prompt[task_id, 1].detach().clone() + self.prefix_MLP_value_layer[task_id](self.prompt[task_id, 1]).detach().clone()
            self.prompt_copy = self.prompt_copy.to(device)
        

        
        
        
