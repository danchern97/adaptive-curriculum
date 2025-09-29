from torch import nn
import torch
import torch.nn.functional as F
from transformers import AutoModel
import sys
from pathlib import Path

# Add searchformer path for imports
searchformer_path = Path(__file__).parent.parent / "searchformer-main"
if str(searchformer_path) not in sys.path:
    sys.path.insert(0, str(searchformer_path))

try:
    from searchformer.transformer.rotary import RoPE
    from searchformer.transformer.model import Attention, RMSLayerNorm
    ROPE_AVAILABLE = True
except ImportError:
    print("Warning: RoPE components not available. Install searchformer or check path.")
    ROPE_AVAILABLE = False

def build_projection(input_size, hidden_size, num_layers=1, use_layernorm=True, dropout=0.1):
    layers = []

    if use_layernorm:
        layers.append(nn.LayerNorm(input_size))

    for i in range(num_layers):
        if i == 0:
            layers.append(nn.Linear(input_size, hidden_size))
        else:
            layers.append(nn.Linear(hidden_size, hidden_size))
        if num_layers > 1 and i < num_layers - 1:
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    
    layers.append(nn.LayerNorm(hidden_size))  

    return nn.Sequential(*layers)

class PureRoPEEncoder(nn.Module):
    """Pure RoPE text encoder without BERT dependencies."""
    
    def __init__(self, vocab_size=50000, hidden_size=768, max_seq_len=2048, rope_dim=128, n_heads=8, n_layers=6):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = hidden_size // n_heads
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Positional embedding (learnable, as backup)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        # RoPE embeddings - initialized with head_dim
        self.rope = RoPE(dim=self.head_dim, max_seq_len=max_seq_len)
        
        # Custom attention layers with RoPE
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': Attention(
                    dim=hidden_size,
                    head_dim=self.head_dim,
                    n_heads=n_heads,
                    dropout=0.1,
                    is_causal=False,
                    rope=self.rope
                ),
                'norm1': RMSLayerNorm((hidden_size,)),
                'norm2': RMSLayerNorm((hidden_size,)),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size * 4, hidden_size),
                    nn.Dropout(0.1)
                )
            }) for _ in range(n_layers)
        ])
        
        print(f"Pure RoPE encoder - Hidden size: {hidden_size}, Head dim: {self.head_dim}, Layers: {n_layers}")
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        B, L = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # [B, L, H]
        
        # Add positional embeddings
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(positions)
        
        # Convert attention mask to the format expected by the custom attention
        attn_mask = None
        if attention_mask is not None:
            # Create causal mask if needed - for now, use no mask for simplicity
            # The searchformer attention expects [batch_size, 1, seq_len, seq_len] mask
            pass
        
        # Apply transformer layers
        for layer in self.layers:
            # Self-attention with RoPE
            attn_out = layer['attention'](x, x, x, mask=attn_mask)
            x = layer['norm1'](x + attn_out)
            
            # Feed-forward with residual connection
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        
        # Pool the sequence (mean pooling with attention mask)
        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(-1).expand(x.size())
            sum_hidden = (x * expanded_mask).sum(dim=1)
            emb_output = sum_hidden / expanded_mask.sum(dim=1).clamp(min=1)
        else:
            emb_output = x.mean(dim=1)
        
        return emb_output

# class TextEncoder(nn.Module):
#     def __init__(self, model_name='bert-base-uncased', lora=False):
#         super().__init__()
#         self.model_name = model_name
#         self.encoder = AutoModel.from_pretrained(
#             model_name,
#             output_hidden_states=True,  # Ensure hidden states are always available
#             # torch_dtype=torch.float16 
#         )
#         if lora:
#             from peft import LoraConfig, get_peft_model
#             lora_config = LoraConfig(
#                 r=8,
#                 lora_alpha=16,
#                 target_modules=["q_proj", "v_proj"],  # You may need to inspect DistilBERT architecture; adjust if needed
#                 lora_dropout=0.1,
#                 bias="none",
#                 task_type="CAUSAL_LM"
#             )
#             self.encoder = get_peft_model(self.encoder, lora_config)
#         self.hidden_size = self.encoder.config.hidden_size
#         print(f"Hidden size: {self.hidden_size}")
        

#     def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Check if token_type_ids is provided
        if token_type_ids is not None:
            outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Debug the output structure
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            last_hidden_state = outputs.hidden_states[-1]
        else:
            # Fallback to the first element which is typically the last_hidden_state
            last_hidden_state = outputs[0]
            
        if 'bert' in self.model_name.lower():
            # Take CLS token 
            emb_output = last_hidden_state[:, 0, :]
        else:
            # Take average of all tokens
            expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            sum_hidden = (last_hidden_state * expanded_mask).sum(dim=1)
            emb_output = sum_hidden / expanded_mask.sum(dim=1) 
        
        return emb_output
       
class ResidualHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, scaling = 'platt',top_k=None):
        super().__init__()
        self.sim_head = RegressionHead(input_size, hidden_size, num_layers=num_layers,top_k=top_k)
            
        self.scaling = scaling
        if self.scaling == 'platt':
            self.res_scale = nn.Parameter(torch.ones(1))
            self.scale = nn.Parameter(torch.ones(1))
        elif self.scaling == 'temperature':
            self.scale = nn.Parameter(torch.ones(1))    
        elif self.scaling == 'group_logit_temp':
            self.mlp        = nn.Sequential(                
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            )
        elif self.scaling == 'plain':
            pass
        else:
            raise ValueError(f"Invalid scaling method: {scaling}")

    def forward(self, q, r, ref_vals,tau):
        base = self.sim_head(q, r, ref_vals, tau)             
        if self.scaling == 'platt':
            out  = torch.sigmoid(self.scale * torch.logit(base.clamp(1e-4, 1-1e-4)) + self.res_scale)
        elif self.scaling == 'temperature':
            out = torch.sigmoid(torch.logit(base.clamp(1e-4, 1-1e-4))/self.scale)
        elif self.scaling == 'group_logit_temp':
            mean_vec = torch.mean(ref_vals, dim=-1, keepdim=True)
            std_vec = torch.std(ref_vals, dim=-1, keepdim=True)
            concat_vec = torch.cat((mean_vec, std_vec), dim=-1)  # (B, 2)
            temp_bias = self.mlp(concat_vec)                     # (B, 2)
            temp = F.softplus(temp_bias[:, 0])                   # (B,)
            bias = torch.tanh(temp_bias[:, 1])                   # (B,)
            out = torch.sigmoid(torch.logit(base.clamp(1e-4, 1-1e-4)) / temp.clamp(1e-2, 10) + bias)
        elif self.scaling == 'plain':
            out = base
        return out
    
class RegressionHead(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers=1,top_k=None):
        super().__init__()
        self.top_k = top_k
        self.query_proj = build_projection(input_size, hidden_size, num_layers=num_layers)
        self.ref_proj = build_projection(input_size, hidden_size, num_layers=num_layers)    

    def forward(self, query_repr, ref_repr, ref_values,tau):
        q_proj = self.query_proj(query_repr)
        r_proj = self.ref_proj(ref_repr)
        scores = torch.bmm(r_proj, q_proj.unsqueeze(-1)).squeeze(-1)/r_proj.size(-1) ** 0.5
        
        if self.top_k is not None and self.top_k < scores.size(1):
            k = self.top_k
            vals, idx = torch.topk(scores, k, dim=1)
            mask = scores.new_full(scores.shape, float('-inf'))
            mask.scatter_(1, idx, vals)
            scores = mask                                   
        weights = scores/tau
        weights -= torch.max(weights, dim=-1, keepdim=True).values
        weights = F.softmax(weights, dim=-1)                         # (B, K)
        pred    = (weights * ref_values).sum(dim=-1)                # (B,)
        return pred

class FewShotRegressor(nn.Module):
    def __init__(self, model_name='bert-base-uncased', method='residual', num_layers=1, 
                 has_embeddings=False, lora=False, scaling='platt', top_k=None, 
                 hidden_size=896, use_rope=False, max_seq_len=2048, rope_dim=128,
                 vocab_size=50000, n_transformer_layers=6):
        super().__init__()
        self.has_embeddings = has_embeddings
        self.use_rope = use_rope
        
        if has_embeddings:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.input_size = config.hidden_size
        else:
            if use_rope and ROPE_AVAILABLE:
                # Use pure RoPE encoder without BERT
                # Ensure hidden_size is divisible by n_heads for proper head_dim
                n_heads = 8  # Default number of heads
                if hidden_size % n_heads != 0:
                    # Adjust hidden_size to be divisible by n_heads
                    hidden_size = ((hidden_size // n_heads) + 1) * n_heads
                    print(f"Adjusted hidden_size to {hidden_size} to be divisible by {n_heads} heads")
                
                self.encoder = PureRoPEEncoder(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    max_seq_len=max_seq_len,
                    n_heads=n_heads,
                    n_layers=n_transformer_layers
                )
                self.input_size = hidden_size
            else:
                # Use BERT-based encoder
                self.encoder = TextEncoder(model_name, lora=lora)
                self.input_size = self.encoder.hidden_size
        
        if method == 'residual':
            self.regressor = ResidualHead(self.input_size, hidden_size, num_layers, scaling=scaling,top_k=top_k)
        else:
            self.regressor = RegressionHead(self.input_size, hidden_size, num_layers,top_k=top_k)
            raise ValueError(f"Invalid method: {method}")

    def forward(self, query_input, ref_input, ref_values,tau=1.0):
        if not self.has_embeddings:
            q_repr = self.encoder(**query_input)
            r_repr = self.encoder(**ref_input)
        else:
            q_repr = query_input
            r_repr = ref_input

        B = q_repr.size(0)
        K = ref_values.size(1)
        r_repr = r_repr.unsqueeze(0).expand(B, K, -1)
                    
        return self.regressor(q_repr, r_repr, ref_values, tau=tau)
    