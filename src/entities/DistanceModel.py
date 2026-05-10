import torch.nn as nn
import torch


class PathEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
    
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        ordered_h_n = h_n.index_select(1, packed.unsorted_indices)
        return ordered_h_n[-1]


class InteractionModule(nn.Module):
    def __init__(self, hidden_dim=64, nhead=4, nlayers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.hidden_dim = hidden_dim

    def forward(self, agents_embs, ids):
        unique_ids, idx = ids.unique(return_inverse=True)
        num_groups = unique_ids.shape[0]
        group_sizes = torch.bincount(idx)
        max_group_size = group_sizes.max()

        group_embs = torch.zeros(num_groups, max_group_size, self.hidden_dim, device=agents_embs.device)
        padding_mask = torch.ones(num_groups, max_group_size, dtype=torch.bool, 
                            device=agents_embs.device)

        for group_idx, group_id in enumerate(unique_ids):
            mask = (ids == group_id)
            group_agents = agents_embs[mask]  # [num_agents_in_group, hidden_dim]
            num_agents = group_agents.size(0)
            group_embs[group_idx, :num_agents] = group_agents
            padding_mask[group_idx, :num_agents] = False

        # group_embs: [num_groups, max_group_size, hidden_dim]
        # padding_mask: [num_groups, max_group_size]
        transformer_output = self.transformer(group_embs, src_key_padding_mask=padding_mask)
        
        masked_output = transformer_output * (~padding_mask).unsqueeze(-1).float()
        valid_lengths = (~padding_mask).sum(dim=1, keepdim=True).float()  
        pooled_outputs = masked_output.sum(dim=1) / torch.clamp(valid_lengths, min=1.0)
        return pooled_outputs


class PredictionHead(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.fc(x).squeeze(1)


class DistanceModel(nn.Module):
    def __init__(self, hidden_dim=128, device=None):
        super().__init__()
        self.path_encoder = PathEncoder(hidden_dim=hidden_dim)
        self.interaction_module = InteractionModule(hidden_dim=hidden_dim)
        self.prediction_head = PredictionHead(hidden_dim=hidden_dim)
        self.device = device

    def forward(self, batch):
        points = batch['points'].to(self.device)
        lengths = batch['lengths'].to(self.device)
        instance_ids = batch['instance_ids'].to(self.device)

        path_embs = self.path_encoder(points, lengths)
        interacted_embs = self.interaction_module(path_embs, instance_ids)
        predictions = self.prediction_head(interacted_embs)
        return predictions