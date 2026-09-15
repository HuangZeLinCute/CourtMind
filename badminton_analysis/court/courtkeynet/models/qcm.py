"""
Quadrilateral Constraint Module
Manuscript: Section 3.5, Equations (30)-(43)
"""
import torch
import torch.nn as nn


class QuadrilateralConstraintModule(nn.Module):
    """
    Encodes geometric properties of quadrilateral into feature space
    """
    def __init__(self, feature_dim, hidden_dims=[64, 128]):
        super().__init__()
        
        # MLP: 10 → 64 → 128 → feature_dim
        layers = []
        dims = [10] + list(hidden_dims) + [feature_dim]
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(inplace=True)
            ])
        
        self.mlp = nn.Sequential(*layers[:-1])  # Remove last ReLU

    def compute_geometric_features(self, kpts):
        """
        Compute 10D geometry vector: [d0,d1,d2,d3, d13,d24, θ0,θ1,θ2,θ3]
        kpts: (B, 4, 2) normalized coordinates
        """
        B = kpts.shape[0]
        
        # Edge lengths (4 sides)
        edges = []
        for i in range(4):
            p1 = kpts[:, i]
            p2 = kpts[:, (i+1) % 4]
            dist = torch.norm(p1 - p2, dim=1, keepdim=True)
            edges.append(dist)
        edges = torch.cat(edges, dim=1)  # (B, 4)
        
        # Diagonal lengths
        d13 = torch.norm(kpts[:,0] - kpts[:,2], dim=1, keepdim=True)
        d24 = torch.norm(kpts[:,1] - kpts[:,3], dim=1, keepdim=True)
        
        ## Internal angles (using dot product)





        # Internal angles (using cosine directly for numerical stability)
        # Using cos(angle) instead of angle avoids acos gradient issues

        angles = []
        for i in range(4):
            prev = kpts[:, (i-1) % 4]
            curr = kpts[:, i]
            next = kpts[:, (i+1) % 4]
            
            v1 = prev - curr
            v2 = next - curr
            
            cos_angle = (v1 * v2).sum(dim=1) / (
                torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + 1e-8
            )
            cos_angle = torch.clamp(cos_angle, -1, 1)

            ## to remove 
            angle = torch.acos(cos_angle).unsqueeze(1)
            angles.append(angle)

            # Use cosine directly (same info, stable gradients)
            #angles.append(cos_angle.unsqueeze(1))





        angles = torch.cat(angles, dim=1)  # (B, 4)
        
        # Concatenate: (B, 10)
        geom = torch.cat([edges, d13, d24, angles], dim=1)
        return geom

    def forward(self, kpt_features, kpts_init):
        """
        kpt_features: (B, 4, C) - per-keypoint features
        kpts_init: (B, 4, 2) - initial keypoint coordinates
        Returns: (B, 4, C) - enhanced features
        """
        # Compute geometric constraints
        geom_vec = self.compute_geometric_features(kpts_init)  # (B, 10)
        
        # Encode to feature space
        geom_emb = self.mlp(geom_vec)  # (B, C)
        
        # Broadcast and add to each keypoint feature
        geom_emb = geom_emb.unsqueeze(1)  # (B, 1, C)
        
        return kpt_features + geom_emb
