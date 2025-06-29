# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f"{m._get_name()}.onnx"
    torch.onnx.export(m, x, f)
    os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim
    ```
"""

from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
    DCNv4_C2f,
    Concat_BiFPN,
    AFEM,
    SimAM,
    SimSPPF,
    C3STR, SPPCSPC,
    ASPP, RFB, LightASPP,
    FDConvBlock,
    ARConvBlock,
    C2f_ContMix,
    C2frepghost,
    SPPFI,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
    ACmix,
    CBAM,
    ECA,
    ShuffleAttention,
    MHSA,
    DCN_v4, DCNv4_Conv,
    ELA,
    SPD,
    VoVGSCSP, VoVGSCSPC, GSConv,
    PConv, PSCConv,

)
from .head import OBB, Classify, Detect, Pose, PoSeg, RTDETRDecoder, Segment, WorldDetect, v10Detect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

from .coordatt import CoordAtt
from .mlla import MLLAttention
from .SwinTransformer import SwinV2_CSPB
from .MobileNetV4 import mobilenetv4_conv_large
from .PPA  import PPA
from .seam import SEAM, MultiSEAM
from .ALSS import ALSS
from .ConvNeXtv2 import convnextv2_atto
from .GAM_Attention import GAM_Attention
from .MLKA import MLKA_Ablation
from .UniRepLKNet import unireplknet_a, unireplknet_f, unireplknet_p, unireplknet_n, unireplknet_t, unireplknet_s, unireplknet_b, unireplknet_l, unireplknet_xl
from .SAFM import SAFMNPP
from .CMRF import CMRF
from .MSAA import MSAA
from .MogaNet import C2f_MultiOGA, ChannelAggregationFFN, MultiOrderGatedAggregation
from .FocalModulation import FocalModulation
from .FDConv_initialversion import FDConv
from .contmix import ContMixBlock
from .DynamGSConv import DynamGSConv

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3k2",
    "SCDown",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "v10Detect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "AConv",
    "ELAN1",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "TorchVision",
    "Index",
    "A2C2f",
    "PoSeg",
    "ACmix",
    "CoordAtt",
    "CBAM",
    "ECA",
    "MLLAttention",
    "ShuffleAttention",
    "MHSA",
    "SwinV2_CSPB",
    "mobilenetv4_conv_large",
    "DCN_v4",
    "DCNv4_Conv",
    "DCNv4_C2f",
    "PPA",
    "SEAM",
    "MultiSEAM",
    "ALSS",
    "convnextv2_atto",
    "Concat_BiFPN",
    "GAM_Attention",
    "MLKA_Ablation",
    "ELA",
    "SPD",
    "VoVGSCSP", "VoVGSCSPC", "GSConv",
    "unireplknet_a", "unireplknet_f", "unireplknet_p", "unireplknet_n", "unireplknet_t", "unireplknet_s", "unireplknet_b", "unireplknet_l", "unireplknet_xl",
    "SAFMNPP",
    "AFEM",
    "CMRF",
    "SimAM",
    "MSAA",
    "C2f_MultiOGA", "ChannelAggregationFFN", "MultiOrderGatedAggregation",  # MogaNet modules
    "FocalModulation",
    "SimSPPF",
    "C3STR", "SPPCSPC",
    "PConv", "PSCConv",
    "ASPP", "RFB", "LightASPP",
    "FDConv",  # FDConv_initialversion
    "FDConvBlock",
    "ARConvBlock",  # ARConv
    "ContMixBlock",
    "C2f_ContMix",
    "C2frepghost",
    "SPPFI",  # SPPFI
    "DynamGSConv",  # DynamGSConv (Dynamic Grouped Convolution
)
