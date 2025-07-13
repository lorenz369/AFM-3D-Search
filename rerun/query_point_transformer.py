import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
import rerun as rr
import numpy as np


# --------------------------------------
# 1. Positional Encoding for (x, y, z)
# --------------------------------------
def positional_encoding(points):
    # points: [N, 3]
    # Simple Fourier Features (harmonic embedding)
    freqs = 2 ** torch.arange(4, dtype=torch.float32).to(points.device)  # [1, 2, 4, 8]
    pos_embeds = [points]
    for freq in freqs:
        pos_embeds.append(torch.sin(points * freq))
        pos_embeds.append(torch.cos(points * freq))
    return torch.cat(pos_embeds, dim=-1)  # [N, 3 * (1 + 2*len(freqs))]


# --------------------------------------
# 2. Point Transformer Backbone
# --------------------------------------

def knn(x, k):
    """ x: [N, 3] → indices of KNN neighbors [N, K] """
    dist = torch.cdist(x, x)  # [N, N]
    _, idx = dist.topk(k=k, largest=False)  # Smallest distances
    return idx  # [N, K]

class PointTransformerLayer(nn.Module):
    def __init__(self, dim, k=16, pos_mlp_hidden_dim=64):
        super().__init__()
        self.k = k
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(dim, dim)

    def forward(self, points, x):
        # points: [N, 3], x: [N, D]
        q = self.linear(x)  # [N, D]

        # Find KNN indices
        idx = knn(points, self.k)  # [N, K]
        N, K = idx.shape
        neighbors = x[idx]  # [N, K, D]
        relative_pos = points.unsqueeze(1) - points[idx]  # [N, K, 3]
        pos_enc = self.pos_mlp(relative_pos)  # [N, K, D]

        attn = F.softmax(self.attn_mlp(q.unsqueeze(1) - neighbors + pos_enc), dim=1)  # [N, K, D]

        out = (attn * (neighbors + pos_enc)).sum(dim=1)  # [N, D]
        return x + self.gamma * out  # Residual connection

class PointTransformerBackbone(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.transformer_layers = nn.ModuleList([
            PointTransformerLayer(dim=hidden_dim, k=16)
            for _ in range(num_layers)
        ])

    def forward(self, points, features):
        pos_embed = positional_encoding(points)
        x = torch.cat([features, pos_embed], dim=-1)
        x = self.input_proj(x)

        for layer in self.transformer_layers:
            x = layer(points, x)

        return x  # [N, hidden_dim]

# --------------------------------------
# 3. Query Decoder (Simple Version)
# --------------------------------------
class Simple3DQueryDecoder(nn.Module):
    def __init__(self, hidden_dim, text_dim=768):  # Assuming ViT-L-14 CLIP
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, features, points, text_embed):
        query = self.text_proj(text_embed).unsqueeze(0)  # [1, 1, D]
        feats = features.unsqueeze(0)                   # [1, N, D]

        attn_output, _ = self.cross_attn(query, feats, feats)  # [1, 1, D]
        attn_output = attn_output.squeeze(0)  # [1, D]

        # Masks
        point_logits = torch.einsum('nd,qd->nq', features, attn_output)  # [N, 1]
        masks = torch.sigmoid(point_logits)

        # Boxes
        boxes = self.box_head(attn_output)  # [1, 6]
        return masks, boxes



# --------------------------------------
# 4. Full Model
# --------------------------------------
class PointTransformerModel(nn.Module):
    def __init__(self, in_features_dim):
        super().__init__()
        self.backbone = PointTransformerBackbone(in_features_dim)
        self.decoder = Simple3DQueryDecoder(hidden_dim=256)

    def forward(self, points, features, text_embed):
        x = self.backbone(points, features)
        masks, boxes = self.decoder(x, points, text_embed)
        return masks, boxes


# --------------------------------------
# 5. Example Usage
# --------------------------------------

data = torch.load('locate-3d/cache/ARKitScenes/42444821.pt')
print(data.keys())

device = "cuda" if torch.cuda.is_available() else "cpu"
points = data['points'].to(device)
features_clip = data['features_clip']
features_dino = data['features_dino']

features = torch.cat([features_clip, features_dino], dim=1) 

clip_dim = data["features_clip"].shape[1]
clip_model_name = "ViT-L/14" if clip_dim == 768 else "ViT-B/32"
clip_model, _ = clip.load(clip_model_name, device=device)

text_query = ["lamp next to bed"]
with torch.no_grad():
    text_tokens = clip.tokenize(text_query).to(device)
    text_embed = clip_model.encode_text(text_tokens).float()
    text_embed = F.normalize(text_embed, dim=-1)

pos_enc_dim = 3 * (1 + 2 * 4)
model = PointTransformerModel(in_features_dim=features.shape[1] + pos_enc_dim).to(device)
masks, boxes = model(points, features, text_embed)

# ---------- 4️⃣ Rerun Visualization ----------
rr.init("afm-query-remote", spawn=True)


points_np = points.cpu().numpy()

# Log all points (gray)
rr.log("world/pointcloud/all_points", rr.Points3D(
    positions=points_np,
    colors=[0.7, 0.7, 0.7],
    radii=0.01
))

# Log predicted mask points (red)
mask = masks[:, 0].detach().cpu().numpy()  # Query 0
threshold = 0.5
selected_points = points_np[mask > threshold]

rr.log("world/pointcloud/query_mask", rr.Points3D(
    positions=selected_points,
    colors=[1.0, 0.0, 0.0],
    radii=0.02
))

# Log bounding box (green)
box = boxes[0].detach().cpu().numpy()
center = box[:3]
size = box[3:6]
extent = size / 2.0
bbox_min = center - extent
bbox_max = center + extent
bbox_corners = np.array([
    bbox_min,
    bbox_max
])

rr.log("world/bbox", rr.Points3D(
    positions=bbox_corners,
    colors=[0.0, 1.0, 0.0],
    radii=0.05
))

# Log the query text
rr.log("query/text", rr.TextDocument(f"Query: '{text_query[0]}'"))