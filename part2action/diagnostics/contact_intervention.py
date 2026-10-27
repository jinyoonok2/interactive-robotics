"""Evaluation-only contact-input interventions on a frozen unified MLP policy."""
import math

import numpy as np
import torch


def rotation_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def install_contact_intervention(model, strength=1.0):
    """Scale the relative geometry INPUT; zero preserves the learned projection bias.

    All candidate actions use the same observation, upstream features, predicted
    contact, and phase. No extra backbone forwards or temporal-state changes.
    Strength one returns the original conditioner output exactly.
    """
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Contact strength must be finite and between zero and one")
    if (not model.use_unified_contact_conditioning or not model.contact_conditioning_enabled
            or model.action_head_type != "mlp" or model.split_action_heads
            or getattr(model, "contact_conditioning_stage", "late") != "late"):
        raise ValueError("Interventions require the enabled unified-contact monolithic MLP")
    if model.contact_geometry_conditioner.mode != "additive":
        raise ValueError("Input-strength interventions currently support the original additive conditioner only")

    def hook(module, inputs, output):
        features = inputs[0]
        relative = output[1]
        strengths = sorted(set([0.0, 0.25, 1.0, float(strength)]))
        projections = {s: module.projection(relative * s) for s in strengths}
        candidates = {}
        for s in strengths:
            projected = projections[s][:, None, :] if features.ndim == 3 else projections[s]
            conditioned = output[0] if s == 1 else features + projected
            candidates[s] = model.action_head(conditioned)[0, 0].detach().float().cpu().numpy()
        base = features.mean(dim=1) if features.ndim == 3 else features
        base_rms = float(base.detach().float().square().mean().sqrt())
        zero = candidates[0.0]
        entries = {}
        for s in strengths:
            action = candidates[s]
            rotation = rotation_matrix(action[3:6]) @ rotation_matrix(zero[3:6]).T
            angle = math.degrees(math.acos(float(np.clip((np.trace(rotation)-1)/2, -1, 1))))
            embedding_rms = float(projections[s].detach().float().square().mean().sqrt())
            signal_rms = float((projections[s]-projections[0.0]).detach().float().square().mean().sqrt())
            entries[str(s)] = {
                "action0": action.tolist(), "xyz_change_from_zero_cm": float(np.linalg.norm(action[:3]-zero[:3])*100),
                "rotation_change_from_zero_deg": angle,
                "projection_rms": embedding_rms, "contact_signal_rms": signal_rms,
                "projection_to_base_rms_ratio": embedding_rms / max(base_rms, 1e-12),
                "contact_signal_to_base_rms_ratio": signal_rms / max(base_rms, 1e-12),
            }
        model.contact_intervention_diagnostic = {
            "executed_strength": strength, "relative_input_norm": float(relative[0].float().norm()),
            "base_pooled_feature_rms": base_rms, "candidates": entries,
        }
        if strength == 1:
            return output
        projected = projections[float(strength)]
        if features.ndim == 3:
            projected = projected[:, None, :]
        return features + projected, relative * strength

    return model.contact_geometry_conditioner.register_forward_hook(hook)
