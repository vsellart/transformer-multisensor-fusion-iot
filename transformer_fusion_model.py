import torch
import torch.nn as nn

# -------------------------------------------------------------------------
# Tensor notation
# -------------------------------------------------------------------------
# B: batch size
# S: number of sensors
# L: byte sequence length
# H: hidden dimension
# M: metadata dimension

# -------------------------------------------------------------------------
# Sensor Byte Embedding
# -------------------------------------------------------------------------

class SensorByteEmbedder(nn.Module):
    def __init__(
        self,
        byte_emb_dim: int = 16,
        meta_dim: int = 1,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.byte_embedding = nn.Embedding(256, byte_emb_dim)

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(byte_emb_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        sensor_bytes: torch.Tensor, # (B*S, L)
        sensor_meta: torch.Tensor, # (B*S, M)
        sensor_byte_mask: torch.Tensor = None, # (B*S, L)
    ) -> torch.Tensor:
        
        BxS, L = sensor_bytes.shape

        byte_repr = self.byte_embedding(sensor_bytes) # (B*S, L, byte_emb_dim)

        meta_repr = self.meta_mlp(sensor_meta) # (B*S, 64)
        meta_repr = meta_repr.unsqueeze(1).expand(-1, L, -1) # (B*S, L, 64)

        byte_repr = torch.cat([byte_repr, meta_repr], dim=-1) # (B*S, L, byte_emb_dim + 64)
        byte_repr = self.fusion(byte_repr) # (B*S, L, H)

        if sensor_byte_mask is not None:
            byte_repr = byte_repr * sensor_byte_mask.unsqueeze(-1).float()

        return byte_repr # (B*S, L, H)
    
# -------------------------------------------------------------------------
# Inter-Sensor Transformer Fusion
# -------------------------------------------------------------------------

class InterSensorTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True,
        )

        self.sensor_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_head = nn.Linear(hidden_dim, 256)

    def forward(
        self,
        sensor_repr: torch.Tensor, # (B, S, L, H)
        sensor_mask: torch.Tensor = None, # (B, S)
        sensor_byte_mask: torch.Tensor = None, # (B, S, L)
    ) -> torch.Tensor:
        
        B, S, L, H = sensor_repr.shape
    
        x = sensor_repr.permute(0, 2, 1, 3).contiguous() # (B, L, S, H)
        x = x.reshape(B * L, S, H) # (B*L, S, H)

        if sensor_mask is not None:
            valid_sensor = sensor_mask.unsqueeze(1).expand(-1, L, -1) # (B, L, S)
        else:
            valid_sensor = torch.ones(B, L, S, device=sensor_repr.device, dtype=torch.bool)

        if sensor_byte_mask is not None:
            valid_byte = sensor_byte_mask.permute(0, 2, 1).bool() # (B, L, S)
            valid = valid_sensor.bool() & valid_byte
        else:
            valid = valid_sensor.bool()

        key_padding_mask = (~valid).reshape(B * L, S) # (B*L, S)
        
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask[all_masked, 0] = False

        x = self.sensor_transformer(x, src_key_padding_mask=key_padding_mask) # (B*L, S, H)                                                              

        mask = valid.reshape(B * L, S).unsqueeze(-1).float() # (B*L, S, 1)
        
        if all_masked.any():
            mask[all_masked, 0, 0] = 1.0

        x = x * mask
        x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0) # (B*L, H)

        x = x.reshape(B, L, H) # (B, L, H)

        logits = self.output_head(x) # (B, L, 256)

        return logits
    
# -------------------------------------------------------------------------
# Byte Reconstruction Model
# -------------------------------------------------------------------------

class ByteReconstructionModel(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 64,
        byte_emb_dim: int = 16,
        meta_dim: int = 1,
        num_heads: int = 4,
        inter_num_layers: int = 1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.sensor_embedder = SensorByteEmbedder(
            byte_emb_dim=byte_emb_dim,
            meta_dim=meta_dim,
            hidden_dim=hidden_dim,
        )

        self.event_model = InterSensorTransformer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=inter_num_layers,
        )

    def forward(
        self,
        sensor_bytes: torch.Tensor, # (B, S, L)
        sensor_meta: torch.Tensor, # (B, S, M)
        sensor_mask: torch.Tensor = None, # (B, S)
        sensor_byte_mask: torch.Tensor = None, # (B, S, L)
    ) -> torch.Tensor:
    
        B, S, L = sensor_bytes.shape

        sensor_bytes = sensor_bytes.reshape(B * S, L)
        sensor_meta = sensor_meta.reshape(B * S, -1)

        if sensor_byte_mask is not None:
            sensor_byte_mask_flat = sensor_byte_mask.reshape(B * S, L)
        else:
            sensor_byte_mask_flat = None

        sensor_repr = self.sensor_embedder(sensor_bytes, sensor_meta, sensor_byte_mask_flat) # (B*S, L, H)
        sensor_repr = sensor_repr.reshape(B, S, L, self.hidden_dim) # (B, S, L, H)

        logits = self.event_model(sensor_repr, sensor_mask=sensor_mask, sensor_byte_mask=sensor_byte_mask) # (B, L, 256)

        return logits