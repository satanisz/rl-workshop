import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class KnapsackTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int = 2, 
                 d_model: int = 64, 
                 nhead: int = 4, 
                 num_layers: int = 3, 
                 dim_feedforward: int = 128, 
                 dropout: float = 0.1):
        super(KnapsackTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        # We can try with or without PE. 
        # Since it's a set problem, PE might actually hurt or be irrelevant, 
        # but sometimes it helps distinguish identical items.
        self.pos_encoder = PositionalEncoding(d_model) 
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask=None, temperature=1.0):
        """
        x: (Batch, Seq_Len, Input_Dim)
        mask: (Batch, Seq_Len) - True for valid items, False for padding (in PyTorch Transformer mask logic mostly works on attention mask)
        temperature: float - used to scale logits before sigmoid (if we were using softmax, here just scaling pre-sigmoid)
        """
        
        # In PyTorch Transformer, src_key_padding_mask should be True for PAD positions.
        # Our dataset mask is True for VALID items. So we invert it.
        if mask is not None:
             src_key_padding_mask = ~mask
        else:
             src_key_padding_mask = None

        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # Transformer forward
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Output head
        # We process the output to get probabilities
        # Shape: (Batch, Seq_Len, 1)
        logits = self.output_head[0](x) # Linear
        logits = self.output_head[1](logits) # ReLU
        logits = self.output_head[2](logits) # Linear (Raw scores)
        
        # Apply temperature
        # Higher temperature -> softer probability (more 0.5s)
        # Lower temperature -> harder probability (closer to 0 or 1)
        logits = logits / temperature
        
        probs = torch.sigmoid(logits) # Final Sigmoid
        
        return probs.squeeze(-1) # (Batch, Seq_Len)
