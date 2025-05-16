import math
import numpy as np
import torch
from .ops import xyxy2xywh

def inner_iou(box1, box2, xywh=True, eps=1e-7, ratio=0.7):
    if not xywh:
        box1, box2 = xyxy2xywh(box1), xyxy2xywh(box2)
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - (w1 * ratio) / 2, x1 + (w1 * ratio) / 2, y1 - (h1 * ratio) / 2, y1 + (h1 * ratio) / 2
    inner_b2_x1, inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - (w2 * ratio) / 2, x2 + (w2 * ratio) / 2, y2 - (h2 * ratio) / 2, y2 + (h2 * ratio) / 2

    # Inner-IoU
    inter = (inner_b1_x2.minimum(inner_b2_x2) - inner_b1_x1.maximum(inner_b2_x1)).clamp_(0) * \
            (inner_b1_y2.minimum(inner_b2_y2) - inner_b1_y1.maximum(inner_b2_y1)).clamp_(0)
    inner_union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
    return inter / inner_union


class WIoU_Scale:
    ''' monotonous: {
            None: origin v1
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
        momentum: The momentum of running mean'''

    iou_mean = 1.
    monotonous = False
    _momentum = 1 - 0.5 ** (1 / 7000)
    _is_train = True

    def __init__(self, iou):
        self.iou = iou
        self._update(self)

    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1


def new_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, SIoU=False, EIoU=False, WIoU=False,
                  MPDIoU=False, ShapeIou=False, PIouV1=False, PIouV2=False, UIoU=False, Inner_iou=False,
                  Focal=False, alpha=1, gamma=0.5, scale=False, eps=1e-7,
                  feat_w=640, feat_h=640, ratio=0.7, ShapeIou_scale=0, PIou_Lambda=1.3, epoch=300):
    """
    计算bboxes iou
    Args:
        box1: predict bboxes
        box2: target bboxes
        xywh: 将bboxes转换为xyxy的形式
        GIoU: 为True时计算GIoU LOSS (yolov8自带)
        DIoU: 为True时计算DIoU LOSS (yolov8自带)
        CIoU: 为True时计算CIoU LOSS (yolov8自带,默认使用)
        SIoU: 为True时计算SIoU LOSS (新增)
        EIoU: 为True时计算EIoU LOSS (新增)
        WIoU: 为True时计算WIoU LOSS (新增)
        MPDIoU: 为True时计算MPDIoU LOSS (新增)
        ShapeIou: 为True时计算ShapeIou LOSS (新增)
        PIouV1/V2: 为True时计算Powerful-IoU LOSS (新增)
        UIoU: 为True时计算Unified-IoU LOSS (新增)
        Inner_iou: 为True时计算InnerIou LOSS (新增)
        Focal: 对IOU损失乘以系数=IOU**gamma,以使回归过程专注于高质量锚框,参考Focal-EIoU Loss
        alpha: AlphaIoU中的alpha参数,默认为1,为1时则为普通的IoU,如果想采用AlphaIoU,论文alpha默认值为3,此时设置CIoU=True则为AlphaCIoU
        gamma: Focal-EIoU中指数系数
        scale: scale为True时,WIoU会乘以一个系数
        eps: 防止除0
        feat_w/h: 特征图大小
        ratio: Inner-IoU对应的是尺度因子,通常取范围为[0.5,1.5],原文中VOC数据集对应的Inner-CIoU和Inner-SIoU设置在[0.7,0.8]之间有较大提升，
        数据集中大目标多则设置<1,小目标多设置>1
        ShapeIou_scale: 为ShapeIou的缩放因子,与数据集中目标的大小相关
        PIou_Lambda: 为Powerful-IoU的超参数
        epoch: 为Unified-IoU的超参数,训练轮数
    Returns:
        iou
    """

    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    if UIoU:
        # Unified-IoU https://arxiv.org/pdf/2408.06636
        # define the center point for scaling
        bb1_xc = x1
        bb1_yc = y1
        bb2_xc = x2
        bb2_yc = y2
        # attenuation mode of hyperparameter "u_ratio"[原链接为ratio]
        linear = True
        cosine = False
        fraction = False
        # assuming that the total training epochs are 300, the "u_ratio" changes from 2 to 0.5
        if linear:
            u_ratio = -0.005 * epoch + 2
        elif cosine:
            u_ratio = 0.75 * math.cos(math.pi * epoch / 300) + 1.25
        elif fraction:
            u_ratio = 200 / (epoch + 100)
        else:
            u_ratio = 0.5
        ww1, hh1, ww2, hh2 = w1 * u_ratio, h1 * u_ratio, w2 * u_ratio, h2 * u_ratio
        bb1_x1, bb1_x2, bb1_y1, bb1_y2 = bb1_xc - (ww1 / 2), bb1_xc + (ww1 / 2), bb1_yc - (hh1 / 2), bb1_yc + (hh1 / 2)
        bb2_x1, bb2_x2, bb2_y1, bb2_y2 = bb2_xc - (ww2 / 2), bb2_xc + (ww2 / 2), bb2_yc - (hh2 / 2), bb2_yc + (hh2 / 2)
        # assign the value back to facilitate subsequent calls
        w1, h1, w2, h2 = ww1, hh1, ww2, hh2
        b1_x1, b1_x2, b1_y1, b1_y2 = bb1_x1, bb1_x2, bb1_y1, bb1_y2
        b2_x1, b2_x2, b2_y1, b2_y2 = bb2_x1, bb2_x2, bb2_y1, bb2_y2
        CIoU = True

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    if scale:
        self = WIoU_Scale(1 - (inter / union))

    # IoU
    # iou = inter / union # ori iou
    iou = torch.pow(inter/(union + eps), alpha) # alpha iou https://arxiv.org/abs/2110.13675
    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or WIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = (cw ** 2 + ch ** 2) ** alpha + eps  # convex diagonal squared
            rho2 = (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4) ** alpha  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))
                if Inner_iou and alpha == 1:
                    iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
                if Focal:
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha)), torch.pow(inter/(union + eps), gamma)  # Focal_CIoU
                else:
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = torch.pow(cw ** 2 + eps, alpha)
                ch2 = torch.pow(ch ** 2 + eps, alpha)
                if Inner_iou and alpha == 1:
                    iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
                if Focal:
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(inter/(union + eps), gamma) # Focal_EIou
                else:
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2) # EIou
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                if Inner_iou and alpha == 1:
                    iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
                if Focal:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha), torch.pow(inter/(union + eps), gamma) # Focal_SIou
                else:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha) # SIou
            elif WIoU and alpha == 1:
                if Inner_iou:
                    iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
                if Focal:
                    raise RuntimeError("WIoU do not support Focal.")
                elif scale:
                    return getattr(WIoU_Scale, '_scaled_loss')(self), (1 - iou) * torch.exp((rho2 / c2)), iou # WIoU https://arxiv.org/abs/2301.10051
                else:
                    return iou, torch.exp((rho2 / c2)) # WIoU v1

            if Inner_iou and alpha == 1:
                iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
            if Focal:
                return iou - rho2 / c2, torch.pow(inter/(union + eps), gamma)  # Focal_DIoU
            else:
                return iou - rho2 / c2  # DIoU

        c_area = cw * ch + eps  # convex area
        if Inner_iou and alpha == 1:
            iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
        if Focal:
            return iou - torch.pow((c_area - union) / c_area + eps, alpha), torch.pow(inter/(union + eps), gamma)  # Focal_GIoU https://arxiv.org/pdf/1902.09630.pdf
        else:
            return iou - torch.pow((c_area - union) / c_area + eps, alpha)  # GIoU https://arxiv.org/pdf/1902.09630.pdf

    elif MPDIoU and alpha == 1:
        # MPDIoU https://arxiv.org/pdf/2307.07662v1
        sq_sum = (feat_w ** 2) + (feat_h ** 2)  # 对应输入image的宽高
        d12 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
        d22 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
        if Inner_iou:
            iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
        if Focal:
            raise RuntimeError("MPDIoU do not support Focal.")
        return iou - (d12 / sq_sum) - (d22 / sq_sum)

    elif ShapeIou and alpha == 1:
        # ShapeIou https://arxiv.org/pdf/2312.17663
        ww = 2 * torch.pow(w2, ShapeIou_scale) / (torch.pow(w2, ShapeIou_scale) + torch.pow(h2, ShapeIou_scale))
        hh = 2 * torch.pow(h2, ShapeIou_scale) / (torch.pow(w2, ShapeIou_scale) + torch.pow(h2, ShapeIou_scale))
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c2 = cw ** 2 + ch ** 2 + eps                            # convex diagonal squared
        center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
        center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        center_distance = hh * center_distance_x + ww * center_distance_y
        distance = center_distance / c2

        omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)

        if Inner_iou:
            iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
        if Focal:
            raise RuntimeError("ShapeIou do not support Focal.")
        return iou - distance - 0.5 * ( shape_cost)

    elif (PIouV1 or PIouV2) and alpha == 1:
        # Powerful-IoU https://www.sciencedirect.com/science/article/abs/pii/S0893608023006640
        dw1 = torch.abs(b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1))
        dw2 = torch.abs(b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1))
        dh1 = torch.abs(b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1))
        dh2 = torch.abs(b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1))
        P = ((dw1 + dw2) / torch.abs(w2) + (dh1 + dh2) / torch.abs(h2)) / 4
        L_v1 = 1 - iou - torch.exp(-P ** 2) + 1

        if Focal:
            raise RuntimeError("PIou do not support Focal.")
        if PIouV1:
            return L_v1
        if PIouV2:
            q = torch.exp(-P)
            x = q * PIou_Lambda
            return 3 * x * torch.exp(-x ** 2) * L_v1

    if Inner_iou and alpha == 1:
        iou = inner_iou(box1, box2, xywh=xywh, ratio=ratio)
    if Focal:
        return iou, torch.pow(inter/(union + eps), gamma)  # Focal_IoU
    else:
        return iou  # IoU

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16, imgsz=640, iou_type='Ciou', Inner_iou=False, Focal=False, Focaler=False, epoch=300, alpha=1):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.iou_type = iou_type  # +++
        self.Inner_iou = Inner_iou  # Inner-IoU
        self.Focal = Focal  # Focal-IoU
        self.imgsz = imgsz  # MPDIoU
        self.Focaler = Focaler  # Focaler-IoU
        self.epoch = epoch  # Unified-IoU
        self.alpha = alpha  # AlphaIoU

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        if self.iou_type == "iou":
            iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, Inner_iou=self.Inner_iou, Focal=self.Focal, alpha=self.alpha)

        elif self.iou_type == "Giou":
            iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, GIoU=True, Inner_iou=self.Inner_iou, Focal=self.Focal, alpha=self.alpha)

        elif self.iou_type == "Diou":
            iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, DIoU=True, Inner_iou=self.Inner_iou, Focal=self.Focal, alpha=self.alpha)

        elif self.iou_type == "Siou":
            iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, SIoU=True, Inner_iou=self.Inner_iou, Focal=self.Focal, alpha=self.alpha)

        elif self.iou_type == "Eiou":
            iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, EIoU=True, Inner_iou=self.Inner_iou, Focal=self.Focal, alpha=self.alpha)

        elif self.iou_type == "Wise-iou":
            iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, WIoU=True, Inner_iou=self.Inner_iou, scale=True)

        elif self.iou_type == "MPDiou":
            # 仅针对正方形image输入
            iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, MPDIoU=True, Inner_iou=self.Inner_iou, feat_w=self.imgsz, feat_h=self.imgsz)

        elif self.iou_type == "Shape-iou":
            iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, ShapeIou=True, Inner_iou=self.Inner_iou, ShapeIou_scale=0)

        elif self.iou_type == "Powerful-iou":
            iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, PIouV1=True, PIouV2=False, PIou_Lambda=1.3)

        elif self.iou_type == "Unified-iou":
            iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=True, UIoU=True, epoch=self.epoch)

        else:
            # 默认Ciou
            iou = new_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True, Inner_iou=self.Inner_iou, Focal=self.Focal, alpha=self.alpha)

        if type(iou) is tuple:
            if len(iou) == 2:
                loss_iou = ((1.0 - iou[0]) * iou[1].detach() * weight).sum() / target_scores_sum
            else:
                loss_iou = (iou[0] * iou[1] * weight).sum() / target_scores_sum

        elif self.iou_type == "Powerful-iou":
            # 已在new_bbox_iou求得1-iou
            loss_iou = (iou * weight).sum() / target_scores_sum

        else:
            if self.Focaler:
                # 引入 Focaler-IoU 回归样本 https://arxiv.org/abs/2401.10525
                # default d=0.00,u=0.95
                d = 0.00
                u = 0.95
                iou = ((iou - d) / (u - d)).clamp(0, 1)

            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

