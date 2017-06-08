# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

import os
import cPickle
import argparse

import numpy as np

import cv2
import caffe

from utils.timer import Timer
from utils.blob import im_list_to_blob
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    
    print "fast_rcnn.test._get_image_blob(): "

    im_orig = im.astype(np.float32, copy=True)
    print "(fast_rcnn.test._get_image_blob) im_orig (copied from origin): ", im_orig
    im_orig -= cfg.PIXEL_MEANS
    print "cfg.PIXEL_MEANS: ", cfg.PIXEL_MEANS
    print "(fast_rcnn.test._get_image_blob) im_orig = im_orig - cfg.PIXEL_MEANS = ", im_orig

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    print "(fast_rcnn.test._get_image_blob) im_shape: ", im_shape
    print "(fast_rcnn.test._get_image_blob) im_size_min: ", im_size_min
    print "(fast_rcnn.test._get_image_blob) im_size_max: ", im_size_max

    processed_ims = []
    im_scale_factors = []

    print "cfg.TEST.SCALES: ", cfg.TEST.SCALES
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        print "image_scale: ", target_size
        print "image_size_min: ", im_size_min
        print "im_scale = float(target_size) / float(im_size_min) = ", im_scale

        # Prevent the biggest axis from being more than MAX_SIZE
        print "cfg.TEST.MAX_SIZE: ", cfg.TEST.MAX_SIZE
        print "image_scale * image_size_max = ", im_scale * im_size_max
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            print "image_scale = (float) cfg.TEST.MAX_SIZE) / float(image_size_max) = ", im_scale

        im = cv2.resize(im_orig, None, 
                                 None, 
                                 fx=im_scale, 
                                 fy=im_scale,
                                 interpolation=cv2.INTER_LINEAR)
        print "image_resized: ", im

        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    print "put the resized images into the blob"
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """

    print "fast_rcnn.test._get_rois_blob(): "

    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))

    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""

    print "fast_rcnn.test._get_blobs(): "

    blobs = {
              'data' : None, 
              'rois' : None
            }

    print "get blobs['data'], image_scale_factors:"
    blobs['data'], im_scale_factors = _get_image_blob(im)

    print "cfg.TEST.HAS_RPN: ", cfg.TEST.HAS_RPN
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)

    return blobs, im_scale_factors


def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

       - config rpn and roi
       - forward network

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
   
    print "fast_rcnn.test.im_detect(): "

    blobs, im_scales = _get_blobs(im, boxes)
    print "(fast_rcnn.test.im_detect) blobs: ", blobs
    print "(fast_rcnn.test.im_detect) im_scales: ", im_scales

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    print "cfg.DEDUP_BOXES: ", cfg.DEDUP_BOXES
    print "cfg.TEST.HAS_RPN: ", cfg.TEST.HAS_RPN
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
  
        print "check duplicated rois and get boxes:"
        print "check cfg.DEBUP_BOXES > 0 and not cfg.TEST.HAS_RPN: "        
        

        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, 
                                        return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]
        print "blobs['rois']: ", blobs['rois']
        print "boxes: ", boxes


    if cfg.TEST.HAS_RPN:

        print "get blobs['im_info] = [blobs['data'].shape[2], blobs['data'].shape[3], image_scale[0]]:"
        print "if cfg.TEST.HAS_RPN: ", cfg.TEST.HAS_RPN

        im_blob = blobs['data']
        blobs['im_info'] = np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32)
        print "image_scales: ", im_scales
        print "blobs['data'].shape: ", im_blob.shape
        print "blobs['im_info']: ", blobs['im_info']

    # reshape network inputs
    print "reshape the network inputs:"
    print "net.blobs['data'].reshape()"
    print "blobs['data'].shape: ", blobs['data'].shape
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        print "cfg.TEST.HAS_RPN == True"
        print "net.blobs['im_info'].reshape()"
        print "blobs[im_info'].shape: ", blobs['im_info'].shape
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        print "cfg.TEST.HAS_RPN == False"
        print "net.blobs['rois'].reshape()"
        print "blobs['rois'].shape: ", blobs['rois'].shape
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    print "network forward: "
    print "prepare 'data':"
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        print "prepare 'im_info':"
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        print "prepare 'rois': "
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    print "net.forward(**forward_kwargs)" 
    blobs_out = net.forward(**forward_kwargs)
    print "blobs_out: ", blobs_out


    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]
        print "if cfg.TEST.HAS_RPN: ", cfg.TEST.HAS_RPN
        print "net.blobs['rois']: ", rois
        print "net.blobs['rois'][:,1:5] = ", rois[:, 1:5]
        print "image_scale[0]: ", im_scales[0]
        print "boxes = rois[:, 1:5] / image_scale[0] = ", boxes


    print "get scores:"
    if cfg.TEST.SVM:
        print "if cfg.TEST.SVM == ", cfg.TEST.SVM
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
        print "scores = net.blobs['cls_score'].data = ", scores
    else:
        print "if cfg.TEST.SVM == ", cfg.TEST.SVM
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']
        print "scores = blobs_out['cls_prob'] = ", scores

   
    print "get pred_boxes"
    if cfg.TEST.BBOX_REG:
        print "if cfg.TEST.BBOX_REG == ", cfg.TEST.BBOX_REG
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
        print "apply bounding-box regression deltas"
        print "box_delta = blobs_out['bbox_pred'] = ", box_deltas
        print "pred_boxes = bbox_transform_inv(boxes, box_deltas) = ", pred_boxes
        print "pred_boxes = clip_boxes(pred_boxes, im.shape) = ", pred_boxes
    else:
        print "if cfg.TEST.BBOX_REG == ", cfg.TEST.BBOX_REG
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        print "simply repeat the boxes, once for each class"
        print "pred_boxes = np.tile(boxes, (1, scores.shape[1])) = ", pred_boxes

    print "cfg.DEDUP_BOXES: ", cfg.DEDUP_BOXES
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        print "if cfg.DEBUP_BOXES > 0 and not cfg.TEST.HAS_RPN"
        print "map scores and predictions back to the original set of boxes"
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes



def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""

    print "(fast_rcnn.test) vis_detections() "

    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()



def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes



def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {
          'im_detect' : Timer(), 
          'misc' : Timer()
    }

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, box_proposals)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                vis_detections(im, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)
