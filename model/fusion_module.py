import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelWiseVotingFusion(nn.Module):
    """
    Pixel-wise voting fusion module for intelligent fusion of multi-stage segmentation outputs
    """
    def __init__(self, num_stages=8, seg_classes=1, fusion_type='weighted_voting'):
        super(PixelWiseVotingFusion, self).__init__()
        
        self.num_stages = num_stages
        self.seg_classes = seg_classes
        self.fusion_type = fusion_type
        
        if fusion_type == 'weighted_voting':
            # Learn weights for each stage
            self.stage_weights = nn.Parameter(torch.ones(num_stages))
            
        elif fusion_type == 'attention_voting':
            # Use attention mechanism to compute weights
            self.attention_conv = nn.Conv2d(num_stages, num_stages, kernel_size=3, padding=1)
            self.attention_softmax = nn.Softmax(dim=1)
            
        elif fusion_type == 'confidence_voting':
            # Confidence-based voting
            self.confidence_net = nn.Sequential(
                nn.Conv2d(num_stages, num_stages // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_stages // 2, num_stages, kernel_size=1),
                nn.Sigmoid()
            )
        
        elif fusion_type == 'adaptive_voting':
            # Adaptive voting mechanism combining multiple strategies
            self.stage_weights = nn.Parameter(torch.ones(num_stages))
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(num_stages, num_stages // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_stages // 2, num_stages, kernel_size=1),
                nn.Sigmoid()
            )
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_stages, num_stages // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_stages // 4, num_stages, kernel_size=1),
                nn.Sigmoid()
            )
    
    def forward(self, stage_outputs):
        """
        Args:
            stage_outputs: List of tensors, each of shape [B, C, H, W]
        Returns:
            fused_output: Tensor of shape [B, C, H, W]
        """
        # Stack all stage outputs
        stacked_outputs = torch.stack(stage_outputs, dim=1)  # [B, num_stages, C, H, W]
        B, num_stages, C, H, W = stacked_outputs.shape
        
        if self.fusion_type == 'simple_voting':
            # Simple voting: binarize probabilities then vote
            binary_outputs = (torch.sigmoid(stacked_outputs) > 0.5).float()
            votes = torch.sum(binary_outputs, dim=1)  # [B, C, H, W]
            fused_output = (votes > num_stages // 2).float()
            
        elif self.fusion_type == 'weighted_voting':
            # Weighted voting
            weights = F.softmax(self.stage_weights, dim=0)  # Normalize weights
            weights = weights.view(1, num_stages, 1, 1, 1)
            weighted_outputs = stacked_outputs * weights
            fused_output = torch.sum(weighted_outputs, dim=1)
            
        elif self.fusion_type == 'attention_voting':
            # Attention-based voting
            # Reshape for attention computation
            reshaped = stacked_outputs.view(B, num_stages, -1)  # [B, num_stages, C*H*W]
            reshaped = reshaped.view(B, num_stages, H, W)  # Assume C=1
            
            attention_weights = self.attention_conv(reshaped)
            attention_weights = self.attention_softmax(attention_weights)
            attention_weights = attention_weights.unsqueeze(2)  # [B, num_stages, 1, H, W]
            
            weighted_outputs = stacked_outputs * attention_weights
            fused_output = torch.sum(weighted_outputs, dim=1)
            
        elif self.fusion_type == 'confidence_voting':
            # Confidence-based voting
            # Calculate confidence for each stage
            probs = torch.sigmoid(stacked_outputs)
            confidence = torch.abs(probs - 0.5) * 2  # Higher confidence when farther from 0.5
            
            # Reshape for confidence network
            confidence_reshaped = confidence.view(B, num_stages, H, W)
            confidence_weights = self.confidence_net(confidence_reshaped)
            confidence_weights = confidence_weights.unsqueeze(2)  # [B, num_stages, 1, H, W]
            
            # Normalize confidence weights
            confidence_weights = F.softmax(confidence_weights, dim=1)
            
            weighted_outputs = stacked_outputs * confidence_weights
            fused_output = torch.sum(weighted_outputs, dim=1)
            
        elif self.fusion_type == 'adaptive_voting':
            # Adaptive voting mechanism
            # 1. Stage weights
            stage_weights = F.softmax(self.stage_weights, dim=0)
            stage_weights = stage_weights.view(1, num_stages, 1, 1, 1)
            
            # 2. Spatial attention
            spatial_input = stacked_outputs.view(B, num_stages, H, W)
            spatial_weights = self.spatial_attention(spatial_input)
            spatial_weights = spatial_weights.unsqueeze(2)  # [B, num_stages, 1, H, W]
            
            # 3. Channel attention
            channel_input = stacked_outputs.view(B, num_stages, H, W)
            channel_weights = self.channel_attention(channel_input)
            channel_weights = channel_weights.view(B, num_stages, 1, 1, 1)
            
            # 4. Combine all weights
            combined_weights = stage_weights * spatial_weights * channel_weights
            combined_weights = F.softmax(combined_weights, dim=1)
            
            weighted_outputs = stacked_outputs * combined_weights
            fused_output = torch.sum(weighted_outputs, dim=1)
        
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        return fused_output


class EnhancedVotingFusion(nn.Module):
    """
    Enhanced voting fusion module with consistency checking and uncertainty estimation
    """
    def __init__(self, num_stages=8, seg_classes=1):
        super(EnhancedVotingFusion, self).__init__()
        
        self.num_stages = num_stages
        self.seg_classes = seg_classes
        
        # Consistency analysis network
        self.consistency_net = nn.Sequential(
            nn.Conv2d(num_stages, num_stages * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_stages * 2, num_stages, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_stages, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.Conv2d(num_stages, num_stages, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_stages, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(num_stages))
        
    def forward(self, stage_outputs):
        """
        Args:
            stage_outputs: List of tensors, each of shape [B, C, H, W]
        Returns:
            fused_output: Tensor of shape [B, C, H, W]
            uncertainty: Tensor of shape [B, 1, H, W] - uncertainty map
        """
        # Stack all stage outputs
        stacked_outputs = torch.stack(stage_outputs, dim=1)  # [B, num_stages, C, H, W]
        B, num_stages, C, H, W = stacked_outputs.shape
        
        # Calculate probabilities
        probs = torch.sigmoid(stacked_outputs)
        
        # Consistency analysis
        probs_reshaped = probs.view(B, num_stages, H, W)
        consistency_score = self.consistency_net(probs_reshaped)  # [B, 1, H, W]
        
        # Uncertainty estimation
        # Use variance as uncertainty indicator
        prob_mean = torch.mean(probs, dim=1, keepdim=True)
        prob_var = torch.var(probs, dim=1, keepdim=True)
        
        uncertainty_input = torch.cat([
            prob_var.view(B, 1, H, W),
            (1 - consistency_score)
        ], dim=1)
        
        uncertainty_input_reshaped = uncertainty_input.repeat(1, num_stages//2, 1, 1)
        uncertainty = self.uncertainty_net(uncertainty_input_reshaped)  # [B, 1, H, W]
        
        # Weighted fusion based on consistency and uncertainty
        weights = F.softmax(self.fusion_weights, dim=0)
        weights = weights.view(1, num_stages, 1, 1, 1)
        
        # Adjust weights based on consistency
        consistency_weights = consistency_score.unsqueeze(1).repeat(1, num_stages, 1, 1, 1)
        adjusted_weights = weights * consistency_weights
        adjusted_weights = F.normalize(adjusted_weights, p=1, dim=1)
        
        # Final fusion
        weighted_outputs = stacked_outputs * adjusted_weights
        fused_output = torch.sum(weighted_outputs, dim=1)
        
        return fused_output, uncertainty


def majority_voting(stage_outputs, threshold=0.5):
    """
    Simple majority voting mechanism
    """
    stacked_outputs = torch.stack(stage_outputs, dim=0)  # [num_stages, B, C, H, W]
    probs = torch.sigmoid(stacked_outputs)
    binary_outputs = (probs > threshold).float()
    votes = torch.sum(binary_outputs, dim=0)  # [B, C, H, W]
    num_stages = len(stage_outputs)
    return (votes > num_stages // 2).float()


def soft_voting(stage_outputs, weights=None):
    """
    Soft voting mechanism based on probability averaging
    """
    stacked_outputs = torch.stack(stage_outputs, dim=0)  # [num_stages, B, C, H, W]
    
    if weights is not None:
        weights = torch.tensor(weights, device=stacked_outputs.device, dtype=stacked_outputs.dtype)
        weights = weights.view(-1, 1, 1, 1, 1)
        weighted_outputs = stacked_outputs * weights
        return torch.sum(weighted_outputs, dim=0) / torch.sum(weights)
    else:
        return torch.mean(stacked_outputs, dim=0)
