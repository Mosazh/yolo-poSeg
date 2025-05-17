import torch
import numpy as np

# reference: https://github.com/dongdonghy/repulsion_loss_pytorch/blob/master/repulsion_loss.py
def IoG(gt_box, pre_box):
    inter_xmin = torch.max(gt_box[:, 0], pre_box[:, 0])
    inter_ymin = torch.max(gt_box[:, 1], pre_box[:, 1])
    inter_xmax = torch.min(gt_box[:, 2], pre_box[:, 2])
    inter_ymax = torch.min(gt_box[:, 3], pre_box[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = ((gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])).clamp(1e-6)
    return I / G

def smooth_ln(x, deta=0.5):
    return torch.where(
        torch.le(x, deta),
        -torch.log(1 - x),
        ((x - deta) / (1 - deta)) - np.log(1 - deta)
    )

def box_iou(box1, box2, x1y1x2y2=True):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box, x1y1x2y2=True):
        # box = nx4
        if x1y1x2y2:
            return (box[2] - box[0]) * (box[3] - box[1])
        else:
            b_x1, b_x2 = box[0] - box[2] / 2, box[0] + box[2] / 2
            b_y1, b_y2 = box[1] - box[3] / 2, box[1] + box[3] / 2
            return (b_x2 - b_x1) * (b_y2 - b_y1)
    area1 = box_area(box1.T, x1y1x2y2)
    area2 = box_area(box2.T, x1y1x2y2)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


# YU 添加了detach，减小了梯度对gpu的占用
def repulsion_loss_torch(pbox, gtbox, deta=0.5, pnms=0.1, gtnms=0.1, x1x2y1y2=False):
    repgt_loss = 0.0
    repbox_loss = 0.0
    pbox = pbox.detach()
    gtbox = gtbox.detach()
    gtbox_cpu = gtbox.cuda().data.cpu().numpy()
    pgiou = box_iou(pbox, gtbox, x1y1x2y2=x1x2y1y2)
    pgiou = torch.where(torch.isnan(pgiou), torch.full_like(pgiou, 0), pgiou)
    pgiou = pgiou.cuda().data.cpu().numpy()
    # if (torch.any(torch.isnan(ppiou)))
    ppiou = box_iou(pbox, pbox, x1y1x2y2=x1x2y1y2)
    ppiou = torch.where(torch.isnan(ppiou), torch.full_like(ppiou, 0), ppiou)
    ppiou = ppiou.cuda().data.cpu().numpy()
    # t1 = time.time()
    len = pgiou.shape[0]
    for j in range(len):
        for z in range(j, len):
            ppiou[j, z] = 0
            # if int(torch.sum(gtbox[j] == gtbox[z])) == 4:
            # if int(torch.sum(gtbox_cpu[j] == gtbox_cpu[z])) == 4:
            # if int(np.sum(gtbox_numpy[j] == gtbox_numpy[z])) == 4:
            if (gtbox_cpu[j][0]==gtbox_cpu[z][0]) and (gtbox_cpu[j][1]==gtbox_cpu[z][1]) and (gtbox_cpu[j][2]==gtbox_cpu[z][2]) and (gtbox_cpu[j][3]==gtbox_cpu[z][3]):
                pgiou[j, z] = 0
                pgiou[z, j] = 0
                ppiou[z, j] = 0

    # t2 = time.time()
    # print("for cycle cost time is: ", t2 - t1, "s")
    pgiou = torch.from_numpy(pgiou).cuda().detach()
    ppiou = torch.from_numpy(ppiou).cuda().detach()
    # repgt
    max_iou, argmax_iou = torch.max(pgiou, 1)
    pg_mask = torch.gt(max_iou, gtnms)
    num_repgt = pg_mask.sum()
    if num_repgt > 0:
        iou_pos = pgiou[pg_mask, :]
        max_iou_sec, argmax_iou_sec = torch.max(iou_pos, 1)
        pbox_sec = pbox[pg_mask, :]
        gtbox_sec = gtbox[argmax_iou_sec, :]
        IOG = IoG(gtbox_sec, pbox_sec)
        repgt_loss = smooth_ln(IOG, deta)
        repgt_loss = repgt_loss.mean()

    # repbox
    pp_mask = torch.gt(ppiou, pnms)  # 防止nms为0, 因为如果为0,那么上面的for循环就没有意义了 [N x N] error
    num_pbox = pp_mask.sum()
    if num_pbox > 0:
        repbox_loss = smooth_ln(ppiou, deta)
        repbox_loss = repbox_loss.mean()
    # mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
    # print(mem)
    torch.cuda.empty_cache()

    return repgt_loss, repbox_loss
