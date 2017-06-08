# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):

    print "lib/fast_rcnn/bbox_transform.bbox_transform_inv()"

    print "boxes.shape: ", boxes.shape
    print "boxes.shape[0]: ", boxes.shape[0]
    print "deltas.shape: ", deltas.shape
    print "deltas.shape[0]: ", deltas.shape[0]
    print "deltas.shape[1]: ", deltas.shape[1]

    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)
    print "boxes = boxes.astype(deltas.dtype, copy=False) = ", boxes

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    print "widths = boxes[:, 2] - boxes[:, 0] + 1.0 = ", widths
    print "heights = boxes[:, 3] - boxes[:, 1] + 1.0 = ", heights
    print "ctr_x = boxes[:, 0] + 0.5 * widths = ", ctr_x
    print "ctr_y = boxes[:, 1] + 0.5 * heights = ", ctr_y

    # indexing with 4 steps
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    print "dx = deltas[:, 0::4] = ", dx
    print "dy = deltas[:, 1::4] = ", dy
    print "dw = deltas[:, 2::4] = ", dw
    print "dh = deltas[:, 3::4] = ", dh

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    print "pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis] = ", pred_ctr_x
    print "pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis] = ", pred_ctr_y
    print "pred_w = np.exp(dw) * widths[:, np.newaxis] = ", pred_w
    print "pred_h = np.exp(dh) * heights[:, np.newaxis] = ", pred_h

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    print "pred_boxes.shape: ", pred_boxes.shape
    print "pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype) = ", pred_boxes

    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    print "# x1"
    print "pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w = ", pred_boxes[:, 0::4]

    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    print "# y1"
    print "pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h = ", pred_boxes[:, 1::4]

    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    print "# x2"
    print "pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w = ", pred_boxes[:, 2::4]

    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    print "# y2"
    print "pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h = ", pred_boxes[:, 3::4]

    return pred_boxes



def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    print "lib/fast_rcnn/bbox_transform.clip_boxes()"
    print "clip boxes to image boundaries"
    print "boxes: [x1, y1, x2, y2]"
    print "boxes: ", boxes
    print "im_shape: ", im_shape

    # x1 >= 0
    print "boxes[:, 0::4]: ", boxes[:, 0::4]
    print "boxes[:, 0::4].shape: ", boxes[:, 0::4].shape
    print "im_shape[1] - 1: ", im_shape[1] - 1
    min_boxes = np.minimum(boxes[:, 0::4], im_shape[1] - 1)
    print "np.minimum(boxes[:, 0::4], im_shape[1] - 1).shape: ", min_boxes.shape
    print "np.minimum(boxes[:, 0::4], im_shape[1] - 1): ", min_boxes
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    print "x1 >= 0"
    print "boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0) = ", boxes[:, 0::4]

    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    print "y1 >= 0"
    print "boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0) = ", boxes[:, 1::4]

    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    print "x2 < im_shape[1]"
    print "boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0) = ", boxes[:, 2::4]

    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    print "y2 < im_shape[0]"
    print "boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0) = ", boxes[:, 3::4]

    return boxes
