
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

"""
    Anchor Generation

      base_anchor

          |

      ratio_anchors  --> _ratio_enum(anchor, ratioanchor, ratios)  --> _whctrs(anchors)
                                                                   --> process ws, hs
          |                                                        --> _mkanchors(ws, hs, x, y)
      
      anchors        --> _scale_enum(anchor, scales)               --> _whctrs(anchors)
                                                                   --> process ws, hs
                                                                   --> _mkanchors(ws, hs, x, y)
"""

import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=16, 
                     ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    print "generate_anchors:"
    print "base_size: ", base_size
    print "ratios: ", ratios
    print "scales: ", scales

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    print "base_size", base_size
    print "base_anchor = np.array([1, 1, base_size, base_size]) - 1 = ", base_anchor

    ratio_anchors = _ratio_enum(base_anchor, ratios)
    print "ratios: ", ratios
    print "ratio_anchors = _ratio_enum(base_anchor, ratios) = ", ratio_anchors
    

    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    print "_whctrs: "

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)

    print "anchor: ", anchor
    print "w = anchor[2] - anchor[0] + 1 = ", w
    print "h = anchor[3] - anchor[1] + 1 = ", h
    print "x_ctr = anchor[0] + 0.5 * (w - 1) = ", x_ctr
    print "y_ctr = anchor[1] + 0.5 * (h - 1) = ", y_ctr

    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    print "_mkanchors:"
    print "ws: ", ws
    print "hs: ", hs
    print "x_ctr: ", x_ctr
    print "y_ctr: ", y_ctr

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]

    print "np.newaxis: ", np.newaxis
    print "ws = ws[:, np.newaxis] = ", ws
    print "np.newaxis: ", np.newaxis
    print "hs = hs[:, np.newaxis] = ", hs

    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    print """
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    """
    print anchors

    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    print "_ratio_enum:"

    w, h, x_ctr, y_ctr = _whctrs(anchor)

    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)

    print "size = w * h = ", size
    print "ratios = ", ratios
    print "size_ratios = size / ratios = ", size_ratios
    print "ws = np.round(np.sqrt(size_ratios)) = ", ws
    print "hs = np.round(ws * ratios) = ", hs
    print "anchors = _mkanchors(ws, hs, x_ctr, y_ctr) = ", anchors

    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    print "_scale_enum:"

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    print "w, h, x_ctr, y_ctr = _whctrs(anchor) = ", w, h, x_ctr, y_ctr
    print "ws = w * scales = ", ws
    print "hs = h * scales = ", hs
    print "anchors = _mkanchors(ws, hs, x_ctr, y_ctr) = ", anchors


    return anchors


if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print time.time() - t
    print a
    from IPython import embed; embed()
