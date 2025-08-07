import torch

class VITAttentionGradRollout:
    def __init__(self, model, discard_ratio=0.9, head_fusion='mean'):
        self.model = model
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        self.attentions = []
        self.gradients = []
        self._register_hooks()
            
    def _register_hooks(self):
        for blk in self.model.vit.encoder.layer:
            blk.attention.attention.register_forward_hook(self._save_attention)
            blk.attention.attention.register_full_backward_hook(self._save_gradient)



    def _save_attention(self, module, input, output):
        self.attentions.append(output[0].detach())


    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    def __call__(self, input_tensor, category_index=None):
        self.attentions = []
        self.gradients = []

        self.model.zero_grad()
        output = self.model(input_tensor)
        logits = output.logits if hasattr(output, 'logits') else output

        if category_index is None:
            return self.compute_rollout_attention()

        # Class-specific gradient
        category_mask = torch.zeros_like(logits)
        category_mask[:, category_index] = 1
        loss = (logits * category_mask).sum()
        loss.backward()

        return self.compute_grad_rollout_attention()

    def compute_rollout_attention(self):
        result = torch.eye(self.attentions[0].size(-1)).to(self.attentions[0].device)
        for attention in self.attentions:
            attn_heads = self._fuse_heads(attention)
            attn_heads = self._discard_low_attention(attn_heads)
            attn_heads = attn_heads + torch.eye(attn_heads.size(-1)).to(attn_heads.device)
            attn_heads = attn_heads / attn_heads.sum(dim=-1, keepdim=True)
            result = torch.matmul(attn_heads, result)
        mask = result[0, 0, 1:]  # exclude CLS token
        mask = mask.reshape(int((mask.numel()) ** 0.5), -1)
        return mask.cpu()

    def compute_grad_rollout_attention(self):
        def _reshape_to_square(mask):
            size = mask.numel()
            root = int(size ** 0.5)
            if root * root == size:
                return mask.reshape(root, root)
            else:
                raise ValueError(f"Cannot reshape mask of size {size} into square. Try visualising it as 1D.")

        result = torch.eye(self.attentions[0].size(-1)).to(self.attentions[0].device)
        for attn, grad in zip(self.attentions, self.gradients):
            weights = grad.mean(dim=1)
            fused = self._fuse_heads(attn * weights)
            fused = self._discard_low_attention(fused)
            fused = fused + torch.eye(fused.size(-1)).to(fused.device)
            fused = fused / fused.sum(dim=-1, keepdim=True)
            result = torch.matmul(fused, result)

        mask = result[0, 1:] if result.dim() == 2 else result[0, 0, 1:]
        return mask.cpu()




    def _fuse_heads(self, attn):
        if self.head_fusion == 'mean':
            return attn.mean(dim=1)
        elif self.head_fusion == 'max':
            return attn.max(dim=1)[0]
        elif self.head_fusion == 'min':
            return attn.min(dim=1)[0]
        else:
            raise ValueError(f"Invalid head_fusion type: {self.head_fusion}")

    def _discard_low_attention(self, attn):
        if self.discard_ratio == 0:
            return attn

        if attn.dim() == 3:  # [B, N, N]
            flat = attn.view(attn.size(0), -1)
            num_discard = int(flat.size(1) * self.discard_ratio)
            threshold, _ = flat.topk(num_discard, dim=1, largest=False)
            threshold = threshold[:, -1].unsqueeze(-1).unsqueeze(-1)
            attn = attn.clone()
            attn[attn < threshold] = 0
        elif attn.dim() == 2:  # [N, N]
            flat = attn.view(-1)
            num_discard = int(flat.size(0) * self.discard_ratio)
            threshold, _ = flat.topk(num_discard, largest=False)
            threshold = threshold[-1]
            attn = attn.clone()
            attn[attn < threshold] = 0
        else:
            raise ValueError(f"Unsupported attention shape: {attn.shape}")
        
        return attn

