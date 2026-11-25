"""Shared per-slot goal predictor and component-masked pose/intent memory."""
import torch
from torch import nn

from models.pose_token_diffusion.contracts import TargetPose, PoseMemory
from models.pose_token_diffusion.blocks.geometry import quaternion_matrix, rotation_matrix, relative_pose
from .contracts import SkillGoalBank
from .targets import SKILLS


class SkillGoalHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_dim
        self.scale = config.translation_scale_m
        self.features = nn.Sequential(nn.Linear(width*4+12,width), nn.GELU(), nn.LayerNorm(width),
                                      nn.Linear(width,width), nn.GELU())
        self.position = nn.Linear(width,3)
        self.rotation = nn.Linear(width,6)
        self.opening = nn.Linear(width,1)
        self.applicability = nn.Linear(width,3)
        self.skill = nn.Linear(width,len(SKILLS))
        nn.init.zeros_(self.position.weight)
        nn.init.zeros_(self.position.bias)
        nn.init.normal_(self.rotation.weight,std=.001)
        with torch.no_grad():
            self.rotation.bias.copy_(torch.tensor([1.,0.,0.,1.,0.,0.]))

    def forward(self, encoded, regions):
        slots = len(regions)
        with torch.autocast(device_type=encoded.tcp_pose.device.type,enabled=False):
            current = quaternion_matrix(encoded.tcp_pose[:,:4])
            center = torch.stack([r.center_world for r in regions],1)
            covariance = torch.stack([r.covariance_world for r in regions],1)
            local = (current[:,None].transpose(-1,-2) @ (center-encoded.tcp_pose[:,None,4:7])[...,None]).squeeze(-1)
            local_cov = current[:,None].transpose(-1,-2) @ covariance.float() @ current[:,None]
        region_features = torch.stack([r.tokens[:,0] for r in regions],1)
        features = torch.cat([region_features, encoded.visual_tokens.mean(1)[:,None].expand(-1,slots,-1),
            encoded.language_tokens[:,1:slots+1], encoded.robot_tokens[:,0,None].expand(-1,slots,-1),
            local/self.scale, local_cov.flatten(2)/self.scale**2],-1).to(region_features.dtype)
        features = self.features(features)
        position_local = local+self.scale*self.position(features).float()
        relative_rotation = rotation_matrix(self.rotation(features).reshape(-1,6)).reshape(-1,slots,3,3)
        with torch.autocast(device_type=current.device.type,enabled=False):
            position = encoded.tcp_pose[:,None,4:7]+(current[:,None] @ position_local[...,None]).squeeze(-1)
            rotation = current[:,None] @ relative_rotation
        return SkillGoalBank(position,rotation,.04*self.opening(features).squeeze(-1).float().sigmoid(),
            torch.ones(*position.shape[:2],3,device=position.device,dtype=torch.bool),
            self.skill(features),self.applicability(features))


class SkillPoseTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_dim
        self.width, self.scale = width,config.translation_scale_m
        self.translation = nn.Sequential(nn.Linear(4,width),nn.GELU(),nn.Linear(width,width))
        self.orientation = nn.Sequential(nn.Linear(7,width),nn.GELU(),nn.Linear(width,width))
        self.intent = nn.Sequential(nn.Linear(2,width),nn.GELU(),nn.Linear(width,width))
        self.skill_embedding = nn.Parameter(torch.randn(len(SKILLS),width)*.02)

    def forward(self, encoded, goal):
        batch = encoded.tcp_pose.shape[0]
        if goal is None:
            return PoseMemory(encoded.visual_tokens.new_zeros(batch,3,self.width),
                torch.zeros(batch,3,device=encoded.tcp_pose.device,dtype=torch.bool),
                encoded.tcp_pose.new_zeros(batch,9))
        goal.validate(batch,1)
        valid = goal.component_valid[:,0]
        confidence = valid.float()
        if goal.applicability_logits is not None:
            confidence = confidence*goal.applicability_logits[:,0].float().sigmoid()
        target = TargetPose(goal.position_world[:,0],goal.rotation_world[:,0],valid[:,:2].any(-1))
        error = relative_pose(encoded.tcp_pose,target)
        error = torch.cat([error[:,:3]*valid[:,0,None],error[:,3:]*valid[:,1,None]],-1)
        dtype = encoded.visual_tokens.dtype
        translation = self.translation(torch.cat([error[:,:3]/self.scale,confidence[:,0,None]],-1).to(dtype))
        orientation = self.orientation(torch.cat([error[:,3:],confidence[:,1,None]],-1).to(dtype))
        intent = self.intent(torch.stack([goal.opening[:,0]/.04,confidence[:,2]],-1).to(dtype))
        intent = intent+goal.skill_logits[:,0].softmax(-1).to(dtype) @ self.skill_embedding.to(dtype)
        tokens = torch.stack([translation,orientation,intent],1)*confidence[:,:,None].to(dtype)
        return PoseMemory(tokens,valid,error)
