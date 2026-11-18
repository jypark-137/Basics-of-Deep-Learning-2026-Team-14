"""ViTPose model access, MLP metadata and unequal-width attention layers."""


from numbers import Integral
import torch
from torch import nn

def get_nuwa_backbone(pose_model):
    """Return the ViT backbone with the attention metadata NuWa expects.

    ViTPose stores transformer blocks in pose_model.backbone.blocks.
    Its Attention uses a local head_dim variable instead of storing it as an
    attribute. Derive that same number from qkv's actual output width.
    Adding this integer attribute does not change any tensor or forward pass.

    This returns the existing backbone, not a copy. Later changes to its
    blocks will therefore also be used by the complete pose model.
    """
    backbone = getattr(pose_model, "backbone", None)
    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("Expected an ordinary ViTPose model with backbone.blocks")

    metadata = []
    for index, block in enumerate(blocks):
        name = "backbone.blocks.{}".format(index)
        mlp = getattr(block, "mlp", None)
        attn = getattr(block, "attn", None)
        if mlp is None or attn is None or hasattr(mlp, "experts"):
            raise ValueError("{} must contain ordinary ViT MLP/attention layers".format(name))
        for module, attribute in ((mlp, "fc1"), (mlp, "fc2"),
                                  (attn, "qkv"), (attn, "proj")):
            if not isinstance(getattr(module, attribute, None), nn.Linear):
                raise ValueError("{}: {} must be a Linear layer".format(name, attribute))

        heads = getattr(attn, "num_heads", 0)
        if not isinstance(heads, int) or heads <= 0:
            raise ValueError("{}: invalid attention head count".format(name))
        if attn.qkv.out_features % (3 * heads):
            raise ValueError("{}: qkv width is incompatible with head count".format(name))
        head_dim = attn.qkv.out_features // (3 * heads)
        if hasattr(attn, "head_dim") and attn.head_dim != head_dim:
            raise ValueError("{}: existing head_dim disagrees with qkv".format(name))
        width = mlp.fc1.in_features
        if not (
            mlp.fc1.out_features == mlp.fc2.in_features
            and mlp.fc2.out_features == width
            and attn.qkv.in_features == width
            and attn.proj.in_features == heads * head_dim
            and attn.proj.out_features == width
        ):
            raise ValueError("{}: inconsistent residual/MLP/attention dimensions".format(name))
        metadata.append((attn, head_dim))

    # Validate every block before adding any metadata.
    for attn, head_dim in metadata:
        attn.head_dim = head_dim
    return backbone


def forward_heatmaps(pose_model, images):
    """Use the existing backbone and pose head, keeping autograd enabled."""
    features = pose_model.backbone(images)
    if getattr(pose_model, "with_neck", False):
        features = pose_model.neck(features)
    heatmaps = pose_model.keypoint_head(features)
    if not torch.is_tensor(heatmaps) or heatmaps.ndim != 4:
        raise ValueError("Expected one heatmap tensor with shape [batch, joints, height, width]")
    return heatmaps


def heatmap_task_loss(pose_model, heatmaps, targets, target_weights):
    """Use the pose head's configured loss, including keypoint visibility.

    For the AP-10K config this is JointsMSELoss(use_target_weight=True).
    This replaces classification cross-entropy as NuWa's original task loss.
    """
    losses = pose_model.keypoint_head.get_loss(heatmaps, targets, target_weights)
    terms = []
    for name, value in losses.items():
        if "loss" not in name:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        terms.extend(item.mean() for item in values if torch.is_tensor(item))
    if not terms:
        raise ValueError("The pose head did not return a tensor loss")
    return sum(terms)


def mlp_layers(model):
    blocks = getattr(getattr(model, "backbone", None), "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("Expected an ordinary ViTPose backbone.blocks")
    layers = []
    for index, block in enumerate(blocks):
        mlp = getattr(block, "mlp", None)
        fc1, fc2 = getattr(mlp, "fc1", None), getattr(mlp, "fc2", None)
        if not isinstance(fc1, nn.Linear) or not isinstance(fc2, nn.Linear):
            raise ValueError("Block {} needs Linear fc1/fc2 layers".format(index))
        if hasattr(fc2, "mask") or hasattr(fc2, "beta"):
            raise ValueError("Use an unmasked model loaded from the base checkpoint")
        if fc1.out_features != fc2.in_features or fc1.in_features != fc2.out_features:
            raise ValueError("Inconsistent MLP dimensions in block {}".format(index))
        layers.append(mlp)
    return layers


def hidden_sizes(model):
    return [mlp.fc1.out_features for mlp in mlp_layers(model)]


def _sizes(values, name, depth):
    if not isinstance(values, (list, tuple)) or len(values) != depth:
        raise ValueError("{} must contain one size per block".format(name))
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
           for value in values):
        raise ValueError("{} must contain positive integer sizes".format(name))
    return [int(value) for value in values]


def _linear_like(layer, in_features, out_features):
    new = nn.Linear(in_features, out_features, bias=layer.bias is not None)
    new = new.to(device=layer.weight.device, dtype=layer.weight.dtype)
    new.weight.requires_grad_(layer.weight.requires_grad)
    if new.bias is not None:
        new.bias.requires_grad_(layer.bias.requires_grad)
    new.train(layer.training)
    return new


class AttentionPruned(nn.Module):
    """NuWa's unequal QK/VO widths using the ordinary ViTPose attention API."""
    def __init__(self, width, num_heads, qk_dim, vo_dim, qkv_bias=True,
                 proj_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads, self.qk_dim, self.vo_dim = num_heads, qk_dim, vo_dim
        self.scale = qk_dim**-0.5
        self.qkv = nn.Linear(width, num_heads * (2 * qk_dim + vo_dim), bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(num_heads * vo_dim, width, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, tokens, _ = x.shape
        q_width, v_width = self.num_heads * self.qk_dim, self.num_heads * self.vo_dim
        q, k, v = self.qkv(x).split((q_width, q_width, v_width), dim=-1)
        q = q.reshape(batch, tokens, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch, tokens, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch, tokens, self.num_heads, self.vo_dim).permute(0, 2, 1, 3)
        attention = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        value = (self.attn_drop(attention) @ v).transpose(1, 2).reshape(batch, tokens, v_width)
        return self.proj_drop(self.proj(value))


def _attention_spec(attn, qk_dim, vo_dim):
    return {"num_heads": int(attn.num_heads), "qk_dim": int(qk_dim), "vo_dim": int(vo_dim),
            "qkv_bias": attn.qkv.bias is not None, "proj_bias": attn.proj.bias is not None,
            "attn_drop": float(getattr(getattr(attn, "attn_drop", None), "p", 0.0)),
            "proj_drop": float(getattr(getattr(attn, "proj_drop", None), "p", 0.0))}


def attention_specs(model):
    result = []
    for block in model.backbone.blocks:
        attn = block.attn
        if isinstance(attn, AttentionPruned):
            result.append(dict(_attention_spec(attn, attn.qk_dim, attn.vo_dim), kind="pruned"))
        else:
            dim = attn.qkv.out_features // (3 * attn.num_heads)
            result.append(dict(_attention_spec(attn, dim, dim), kind="dense"))
    return result
