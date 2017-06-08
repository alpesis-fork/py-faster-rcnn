# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms

DEBUG = False


class ProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._feat_stride = layer_params['feat_stride']
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        print "(ProposalLayer) self._anchors: ", self._anchors
        print "(ProposalLayer) self._anchors.shape: ", self._anchors.shape
        self._num_anchors = self._anchors.shape[0]

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)


    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        print "(ProposalLayer) forward:"
        print "(ProposalLayer) bottom ", bottom

        assert bottom[0].data.shape[0] == 1, 'Only single item batches are supported'
        print "(ProposalLayer) bottom[0].data.shape[0]: ", bottom[0].data.shape[0]
        print "(ProposalLayer) bottom[0].data -> scores: ", bottom[0].data
        print "(ProposalLayer) bottom[1].data -> bbox_deltas: ", bottom[1].data
        print "(ProposalLayer) bottom[2].data -> im_info: ", bottom[2].data

        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE
        print "(ProposalLayer) cfg_key: ", cfg_key
        print "(ProposalLayer) pre_nms_topN: ", pre_nms_topN
        print "(ProposalLayer) post_nms_topN: ", post_nms_topN
        print "(ProposalLayer) nms_thresh: ", nms_thresh
        print "(ProposalLayer) min_size: ", min_size

        # the first set of _num_anchors channels are background probs
        # the second set are the frontground probs, which we want
        scores = bottom[0].data[:, self._num_anchors:, :, :]
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]
        print "(ProposalLayer) scores.shape: ", scores.shape
        print "(ProposalLayer) bbox_deltas.shape: ", bbox_deltas.shape
        print "(ProposalLayer) im_info.shape: ", im_info.shape
        print "(ProposalLayer) self._num_anchors: ", self._num_anchors
        print "(ProposalLayer) scores = bottom[0].data[:, self._num_anchors:, :, :] = ", scores
        print "(ProposalLayer) bbox_deltas = bottom[1].data = ", bbox_deltas
        print "(ProposalLayer) im_info = bottom[2].data[0, :] = ", im_info

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]
        print "(ProposalLayer) scores.shape: ", scores.shape
        print "(ProposalLayer) height: ", height
        print "(ProposalLayer) width: ", width

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        print "(ProposalLayer) self._feat_stride: ", self._feat_stride
        print "(ProposalLayer) shift_x: ", shift_x
        print "(ProposalLayer) shift_y: ", shift_y
        print "(ProposalLayer) shift_x.ravel(): ", shift_x.ravel()
        print "(ProposalLayer) shift_y.ravel(): ", shift_y.ravel()
        print "(ProposalLayer) shifts: ", shifts
        print "(ProposalLayer) shifts.shape: ", shifts.shape

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        print "(ProposalLayer) A: ", A
        print "(ProposalLayer) K: ", K
        print "(ProposalLayer) self._anchors: ", self._anchors
        print "(ProposalLayer) self._anchors.reshape((1, A, 4)): ", self._anchors.reshape((1, A, 4))
        print "(ProposalLayer) shifts.reshape((1, K, 4)): ", shifts.reshape((1, K, 4))
        print "(ProposalLayer) shifts.reshape((1, K, 4)).transpose((1, 0, 2)): ", shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        print "(ProposalLayer) anchors: ", anchors
        print "(ProposalLayer) anchors.shape: ", anchors.shape
        anchors = anchors.reshape((K * A, 4))
        print "(ProposalLayer) anchors (final): ", anchors
        print "(ProposalLayer) anchors.shape: ", anchors.shape

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        print "(ProposalLayer) bbox_deltas.shape: ", bbox_deltas.shape
        print "(ProposalLayer) bbox_deltas: ", bbox_deltas
        print "(ProposalLayer) bbox_deltas[0].shape: ", bbox_deltas[0].shape
        print "(ProposalLayer) bbox_deltas[0]: ", bbox_deltas[0]
        print "(ProposalLayer) bbox_deltas.transpose((0, 2, 3, 1)): ", bbox_deltas.transpose((0, 2, 3, 1))
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
        print "(ProposalLayer) bbox_deltas_reshaped.shape: ", bbox_deltas.shape
        print "(ProposalLayer) bbox_deltas_reshaped: ", bbox_deltas

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        print "(ProposalLayer) scores.shape: ", scores.shape
        print "(ProposalLayer) scores: ", scores
        print "(ProposalLayer) scores.transpose((0, 2, 3, 1)): ", scores.transpose((0, 2, 3, 1))
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
        print "(ProposalLayer) scores.shape: ", scores.shape
        print "(ProposalLayer) scores: ", scores

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)
        print "(ProposalLayer) proposals.shape: ", proposals.shape
        print "(ProposalLayer) proposals: ", proposals

        # 2. clip predicted boxes to image
        print "(ProposalLayer) im_info.shape: ", im_info.shape
        print "(ProposalLayer) im_info: ", im_info
        print "(ProposalLayer) im_info[:2]: ", im_info[:2]
        # im_info: [600, 800, ratio]
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        print "(ProposalLayer) min_size: ", min_size
        print "(ProposalLayer) im_info[2]: ", im_info[2]
        keep = _filter_boxes(proposals, min_size * im_info[2])
        print "(ProposalLayer) keep: ", keep
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        print "(ProposalLayer) nms_thresh: ", nms_thresh
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        print "(ProposalLayer) keep: ", keep
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        # [Optional] output scores blob
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
            top[1].data[...] = scores


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    
    print "(ProposalLayer) _filter_boxes:"

    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1

    print "(ProposalLayer) keep_full = ", np.where((ws >= min_size) & (hs >= min_size))
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
