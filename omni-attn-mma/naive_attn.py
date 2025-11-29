import torch
import torch.nn.functional as F
import math

def naive_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float = None
) -> torch.Tensor:
    """
    Naive attention implementation with mask.
    
    Args:
        Q: [B, H, N, head_dim]
        K: [B, H, N, head_dim]
        V: [B, H, N, head_dim]
        attn_mask: [B, N, N] or [B, H, N, N] - True means can attend, False means masked
        scale: scaling factor (default: 1/sqrt(head_dim))
    
    Returns:
        output: [B, H, N, head_dim]
    """
    B, H, N, head_dim = Q.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Ensure mask is [B, H, N, N]
    if attn_mask.ndim == 3:
        attn_mask = attn_mask.unsqueeze(1).expand(B, H, N, N)
    
    # Compute attention scores: [B, H, N, N]
    scores = torch.einsum('bhnd,bhmd->bhnm', Q, K) * scale
    
    # Apply mask: masked positions become -inf
    scores = scores.masked_fill(~attn_mask, float('-inf'))
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply to values
    output = torch.einsum('bhnm,bhmd->bhnd', attn_weights, V)
    
    return output
