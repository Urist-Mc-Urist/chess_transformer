import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class StochasticDepth(nn.Module):
    def __init__(self, p=0.8):
        super().__init__()
        self.p = p

    def forward(self, x, residual):
        if self.training:
            if torch.rand(1).item() < self.p:
                return x + residual
            else:
                return x
        else:
            return x + self.p * residual

class AdvancedTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, stoch_depth_p=0.8):
        super().__init__()
        dim_feedforward = 4 * d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.stoch_depth = StochasticDepth(stoch_depth_p)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        # x shape: (seq_len, batch_size, d_model)
        norm_x = self.norm1(x)

        # Convert boolean mask to float mask
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.float().masked_fill(
                src_key_padding_mask, float('-inf')).masked_fill(~src_key_padding_mask, float(0.0))

        attn_output, _ = self.self_attn(norm_x, norm_x, norm_x, 
                                        attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
        x = self.stoch_depth(x, self.dropout(attn_output))
        
        norm_x = self.norm2(x)
        ff_output = self.ff(norm_x)
        x = self.stoch_depth(x, self.dropout(ff_output))
        return x

class ChessTransformer(nn.Module):
    def __init__(self, num_layers=64, d_model=1024, nhead=8, dropout=0.1, stoch_depth_p=0.9, num_tokens=2066, pad_token_id=2064):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            AdvancedTransformerLayer(d_model, nhead, dropout, stoch_depth_p)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, num_tokens)
        self.d_model = d_model
        self.padding_idx = pad_token_id

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def pad_sequences(self, sequences):
        padding_value = self.padding_idx
        max_len = max(len(seq) for seq in sequences)
        padded_seqs = [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]
        return torch.LongTensor(padded_seqs)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.size()
        
        # Create padding mask
        padding_mask = (x == self.padding_idx)
        
        # Create causal mask
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Embed and add positional encoding
        x = self.embedding(x).transpose(0, 1) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Pass through each layer
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=padding_mask)
        
        x = self.norm(x)
        output = self.output(x.transpose(0, 1))
        
        return output

def winning_moves_loss(output, ground_truth, win_labels, pad_token_id=2064, start_token_id=2065):
    """
    Compute the loss only for the winning moves of white and black.
    """
    output = output.cuda()
    ground_truth = ground_truth.cuda()
    win_labels = win_labels.cuda()

    batch_size, seq_len, num_tokens = output.shape
    
    # Shift the ground truth to align with the output predictions
    ground_truth_shifted = ground_truth[:, 1:].contiguous()
    output_shifted = output[:, :-1, :].contiguous()
    
    # Flatten the output and ground truth for easier masking
    output_flat = output_shifted.view(-1, num_tokens)
    ground_truth_flat = ground_truth_shifted.view(-1)
    
    # Apply log softmax to the flattened output
    output_log_softmax = F.log_softmax(output_flat, dim=-1)
    
    # Repeat win_labels for each move in the sequence
    win_labels_expanded = win_labels.unsqueeze(1).repeat(1, seq_len - 1).view(-1)
    
    # Create a mask for the winning moves
    move_indices = torch.arange(seq_len - 1, device=output.device).unsqueeze(0).repeat(batch_size, 1).view(-1)
    white_win_mask = (win_labels_expanded == 1) & (move_indices % 2 == 0)
    black_win_mask = (win_labels_expanded == 0) & (move_indices % 2 == 1)
  
    # Combine the masks
    selected_moves_mask = (white_win_mask | black_win_mask) & (ground_truth_flat != pad_token_id) & (ground_truth_flat != start_token_id)

    # Calculate the negative log-likelihood loss only for the selected moves
    loss = F.nll_loss(output_log_softmax, ground_truth_flat, reduction='none')

    loss = loss * selected_moves_mask.float()
    
    # Average the loss over the selected moves
    selected_moves_count = selected_moves_mask.float().sum()
    if selected_moves_count > 0:
        loss = loss.sum() / selected_moves_count
    else:
        loss = loss.sum()  # If no moves are selected, return 0 loss
    
    return loss

def all_moves_loss(output, ground_truth, pad_token_id=2064, start_token_id=2065):
    """
    Compute the loss for all valid moves in the sequence, excluding start and padding tokens.
    """
    batch_size, seq_len, num_tokens = output.shape

    output = output.cuda()
    ground_truth = ground_truth.cuda()
    
    # Shift the output and ground truth to align them
    output_shifted = output[:, :-1, :].contiguous()
    ground_truth_shifted = ground_truth[:, 1:].contiguous()
    
    # Flatten the shifted output and ground truth
    output_flat = output_shifted.view(-1, num_tokens)
    ground_truth_flat = ground_truth_shifted.view(-1)

    # Apply log softmax to the flattened output
    output_log_softmax = F.log_softmax(output_flat, dim=-1)

    # Create a mask for all valid moves (excluding padding and start tokens)
    valid_moves_mask = ((ground_truth_flat != pad_token_id) & 
                        (ground_truth_flat != start_token_id))

    # Calculate the negative log-likelihood loss for all moves
    loss = F.nll_loss(output_log_softmax, ground_truth_flat, reduction='none')
    
    # Apply the mask to exclude padding and start tokens
    loss = loss * valid_moves_mask.float()
    
    # Average the loss over all valid moves
    valid_moves_count = valid_moves_mask.float().sum()
    if valid_moves_count > 0:
        loss = loss.sum() / valid_moves_count
    else:
        loss = loss.sum()  # If no valid moves, return 0 loss
    
    return loss

def weighted_chess_loss(output, ground_truth, win_labels, winning_weight=1.0, losing_weight=0.1, pad_token_id=2064, start_token_id=2065):
    """
    Compute a weighted loss for all moves, with higher weight for winning moves.
    """
    output = output.cuda()
    ground_truth = ground_truth.cuda()
    win_labels = win_labels.cuda()

    batch_size, seq_len, num_tokens = output.shape
    
    # Shift the ground truth to align with the output predictions
    ground_truth_shifted = ground_truth[:, 1:].contiguous()
    output_shifted = output[:, :-1, :].contiguous()
    
    # Flatten the output and ground truth for easier masking
    output_flat = output_shifted.view(-1, num_tokens)
    ground_truth_flat = ground_truth_shifted.view(-1)
    
    # Apply log softmax to the flattened output
    output_log_softmax = F.log_softmax(output_flat, dim=-1)
    
    # Repeat win_labels for each move in the sequence
    win_labels_expanded = win_labels.unsqueeze(1).repeat(1, seq_len - 1).view(-1)
    
    # Create masks for winning and losing moves
    move_indices = torch.arange(seq_len - 1, device=output.device).unsqueeze(0).repeat(batch_size, 1).view(-1)
    white_win_mask = (win_labels_expanded == 1) & (move_indices % 2 == 0)
    black_win_mask = (win_labels_expanded == 0) & (move_indices % 2 == 1)
    winning_moves_mask = white_win_mask | black_win_mask
    
    # Create a mask for all valid moves (excluding padding and start tokens)
    valid_moves_mask = (ground_truth_flat != pad_token_id) & (ground_truth_flat != start_token_id)
    
    # Calculate the negative log-likelihood loss for all valid moves
    loss = F.nll_loss(output_log_softmax, ground_truth_flat, reduction='none')
    
    # Apply weights based on whether the move is winning or losing
    weights = torch.where(winning_moves_mask & valid_moves_mask, winning_weight, losing_weight)
    
    # Apply the weights and the valid moves mask to the loss
    weighted_loss = loss * weights * valid_moves_mask.float()
    
    # Average the loss over all valid moves
    valid_moves_count = valid_moves_mask.float().sum()
    if valid_moves_count > 0:
        avg_loss = weighted_loss.sum() / valid_moves_count
    else:
        avg_loss = weighted_loss.sum()  # If no valid moves, return 0 loss
    
    return avg_loss