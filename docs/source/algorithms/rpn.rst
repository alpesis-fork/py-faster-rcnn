RPN
==============================================================================


Key Concepts
------------------------------


Mathematical Equations
------------------------------

Process Steps
------------------------------

Anchors:

refpath: ``lib/rpn/generate_anchors.py``

::

    1. base_anchor = [1, 1, base_size, base_size] - 1

        e.g. base_size = 16
             base_anchor = [1, 1, 16, 16] - 1 = [0, 0, 15, 15]

    2. ratio_anchors

        ratio_anchors = _ratio_enum(base_anchor, ratios)

        2.1. w, h, x_ctr, y_ctr = _whctrs(base_anchor)

            w = anchor[2] - anchor[0] + 1
            h = anchor[3] - anchor[1] + 1
            x_ctr = anchor[0] + 0.5 * (w - 1)
            y_ctr = anchor[1] + 0.5 * (h - 1)

            e.g. 
                 base_anchor = [0, 0, 15, 15]
                 w = 15 - 0 + 1 = 16
                 h = 15 - 0 + 1 = 16
                 x_ctr = 0 + 0.5 * (16 - 1) = 7.5
                 y_ctr = 0 + 0.5 * (16 - 1) = 7.5

        2.2. ws, hs

            ws = np.round(np.sqrt((w * h) / ratios))
            hs = np.round(ws * ratios)

        2.3. anchors = _mkanchors(ws, hs, x_ctr, y_ctr)

            ws = ws[:, np.newaxis]
            hs = hs[:, np.newaxis]

            anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                                 y_ctr - 0.5 * (hs - 1),
                                 x_ctr + 0.5 * (ws - 1),
                                 y_ctr + 0.5 * (hs - 1)))


    3. anchors 
    
        anchors = np.vstack([_scale_enum(ratio_anchors[i,:], scales) for i in xrange(ratio_anchors.shape[0])])

        3.1. _scale_enum()

            x, h, x_ctr, y_ctr = _whctrs()
            ws = w * scales
            hs = h * scales
            anchors = _mkanchors(ws, hs, x_ctr, y_ctr)

Forward:

Steps:

1. generating proposals from bbox_deltas by anchors and shifts

- generate anchors (9, 4)
- generate shifts (1700, 4)
- generate shifted_anchors (9 * 1700, 4) = (17100, 4)
- bbox_deltas: (transpose -> reshape) (1, 4 * 9, 38, 50) -> (1 * 38 * 50 * 4 * 9, 4) = (17100, 4)
- scores: (transpose -> reshape) (1, 9, 38, 50) ->  (1 * 50 * 38 * 9, 1) = (17100, 1)
- proposals: bbox_transform_inv(shifted_anchors, bbox_deltas) (17100, 4)

2. clipping predicted boxes to an image

- proposals: clip_boxes(proposals, im_info[:2])

3. removing predicted boxes with eighter height/width < threshold

4. sorting all (proposals, scores) pairs by score from highest to lowest

5. taking top pre_nms_topN

6. applying nms

7. taking after_nms_topN

8. return the top proposals (RoIs top)

9. output rois blob

10. output scores blob


refpath: ``rpn/proposal_layer.forward()``

1. generating proposals

::

   1. get scores, bbox_deltas, im_info

        scores = bottom[0].data[:, self._num_anchors:, :, :]
        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0,:] 

        scores.shape: (,, height, width), (1, 9, 38, 50)
        bbox_deltas.shape: (,, height, width), (1, 36, 38, 50)
        im_info.shape: (3,)


    2. generate proposals from bbox deltas and shifted anchors

        height, width = scores.shape[-2:]


    3. enumerate all shifts

        shifts = [shift_x, shift_y, shift_x, shift_y]

        - shift_x = [0 * feat_stride, 1 * feat_stride, 2 * feat_stride, ..., (width - 1) * feat_stride]
        - shift_y = [0 * feat_stride, 1 * feat_stride, 2 * feat_stride, ..., (height - 1) * feat_stride]
        
        For instance:
        
           height: 38
           width: 50
           feat_stride: 16

           shift_x: [0*16, 1*16, 2*16, ..., 49*16] = [0, 16, 32, ..., 784]
           shift_y: [0*16, 1*16, 2*16, ..., 37*16] = [0, 16, 32, ..., 592]

           meshgrid(shift_x, shift_y):
           
               - shift_x: shape (38, 50) 

                  [[0, 16, 32, ..., 784],
                   [0, 16, 32, ..., 784],
                   ...
                   [0, 16, 32, ..., 784]]

               - shift_y: shape (38, 50)

                  [[  0,   0,   0, ...,   0],
                   [ 16,  16,  16, ...,  16],
                   ...
                   [592, 592, 592, ..., 592]]

            shifts: [shift_x, shift_y, shift_x, shift_y]

                  [[0, 16, 32, ..., 784, 0, 16, 32, ..., 784, ..., 0, 16, 32, ..., 784],
                   [0, 0, 0, ..., 0, 16, 16, 16, ..., 16, ..., 592, 592, 592, ..., 592],
                   [0, 16, 32, ..., 784, 0, 16, 32, ..., 784, ..., 0, 16, 32, ..., 784],
                   [0, 0, 0, ..., 0, 16, 16, 16, ..., 16, ..., 592, 592, 592, ..., 592]]

            shifts.transpose(): shape (38 * 50, 4)

                  [[  0,   0,   0,   0],
                   [ 16,   0,  16,   0],
                   [ 32,   0,  32,   0],
                   ...
                   [784, 592, 784, 592]]


    4. enumerate all shifted anchors

        anchors = anchors + shifts
        shifted_anchors = anchors ( # of shifts * # of anchors, 4)

        - anchors -> blob: anchors.shape (1, 9, 4) = 1 * 4 * 9
        - shifts -> blob: shifts.shape (1, 1900, 4) = 1 * 4 * 1900
        - shifts.transpose: 

        anchors: 

               [[ -84,  -40,   99,   55],
                [-176,  -88,  191,  103],
                [-360, -184,  375,  199],
                [ -56,  -56,   71,   71],
                [-120, -120,  135,  135],
                [-248, -248,  263,  263],
                [ -36,  -80,   51,   95],
                [ -80, -168,   95,  183],
                [-168, -344,  183,  359]] 

        anchors.reshape: (1, 9, 4)

               [[[ -84,  -40,   99,   55],
                 [-176,  -88,  191,  103],
                 [-360, -184,  375,  199],
                 [ -56,  -56,   71,   71],
                 [-120, -120,  135,  135],
                 [-248, -248,  263,  263],
                 [ -36,  -80,   51,   95],
                 [ -80, -168,   95,  183],
                 [-168, -344,  183,  359]]]

        shifts: (1900, 4)

               [[  0,   0,   0,   0],
                [ 16,   0,  16,   0],
                [ 32,   0,  32,   0],
                ..., 
                [752, 592, 752, 592],
                [768, 592, 768, 592],
                [784, 592, 784, 592]]


        shifts.reshape: (1, 1900, 4)

               [[[  0,   0,   0,   0],
                 [ 16,   0,  16,   0],
                 [ 32,   0,  32,   0],
                 ..., 
                 [752, 592, 752, 592],
                 [768, 592, 768, 592],
                 [784, 592, 784, 592]]]

        shifts.transpose((1, 0, 2)): (1, 1900, 4)

               [[[  0,   0,   0,   0]],
                [[ 16,   0,  16,   0]],
                [[ 32,   0,  32,   0]],
                ..., 
                [[752, 592, 752, 592]],
                [[768, 592, 768, 592]],
                [[784, 592, 784, 592]]]

        shifted_anchors (1900, 9, 4) = anchors (1, 9, 4) + shifts (1, 1900, 4)

               [[anchors(1, 9, 4) + shift[1]],
                [anchors(1, 9, 4) + shift[2]],
                [anchors(1, 9, 4) + shift[3]],
                ...
                [anchors(1, 9, 4) + shift[1900]]]

        shifted_anchors_transposed = shifted_anchors.reshape((# of shifts * # of anchors, 4))
         - shifted_anchors_transposed: (1900 * 9, 4) = (17100, 4)
         - shifted_anchors: (1900, 9, 4)
         - # of shifts: 1900
         - # of anchors: 9
        
               [[ shifted_anchor[1]     ],
                [ shifted_anchor[2]     ],
                [ shifted_anchor[3]     ],
                ...
                [ shifted_anchor[1900]  ],
                [ shifted_anchor[1911]  ],
                ...
                [ shifted_anchor[17100] ]]

    5. (bbox_deltas) transpose and reshape predicted bbox

        5.1. transpose

          bbox_deltas: shape (1, 4 * n_anchors, height, width) -> (1, height, width, 4 * n_anchors)
          - bbox_deltas.shape(0, 1, 2, 3) -> bbox_deltas.reshape(0, 2, 3, 1)

          - bbox_deltas.shape: (1, 36, 38, 50)
          - bbox_deltas[0].shape: (36, 38, 50)
          - (bbox_deltas) anchors: 4 * n_anchors = 4 * 9 = 36
          - (bbox_deltas) height: 38
          - (bbox_deltas) width: 50
         

        5.2. reshape

           bbox_deltas_transposed: (1, height, width, 4 * n_anchors)
           -> reshape: (1 * height * width * 4 * n_anchors, 4)

    6. (scores) transpose -> reshape

        6.1. transpose

            scores: (1, n_anchors, height, width) -> (1, height, width, n_anchors)

            - scores.shape: (1, 9, 38, 50)

        6.2. reshape

            scores: (1, height, width, n_anchors) -> (1 * height * width * n_anchors, 1)

            - scores_reshaped.shape: (1 * 50 * 38 * 9, 1) = (17100, 1)


    7. anchors -> proposals: bbox transformations

        proposals = bbox_transform_inv(anchors, bbox_deltas)

        if boxes.shape[0] == 0:
            return array(0, 4)

        widths' = widths - x_centroids + 1.0
        heights' = heights - y_centroids + 1.0
        x_centroids' = x_centroids + 0.5 * widths
        y_centroids' = y_centroids + 0.5 * heights

        For instance:

            widths = boxes[:, 2] - boxes[:, 0] + 1.0
            heights = boxes[:, 3] - boxes[:, 1] + 1.0
            x_centroid = boxes[:, 0] + 0.5 * widths
            y_centroid = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]

        predicted_x_centroids = dx * widths'.T + x_centroid.T
        predicted_y_centroids = dy * heights'.T + y_cetnroids.T
        # np.exp(x) = e^x
        predicted_widths = np.exp(dw) * widths.T
        predicted_heights = np.exp(dh) * heights.T

        predicted_boxes.shape: (17100, 4)
        # x1
        predicted_boxes[:, 0::4] = predicted_x_centroids - 0.5 * predicted_widths
        # y1
        predicted_boxes[:, 1::4] = predicted_y_centroids - 0.5 * predicted_heights
        # x2
        predicted_boxes[:, 2::4] = predicted_x_centroids + 0.5 * predicted_widths
        # y2
        predicted_boxes[:, 3::4] = predicted_y_centroids + 0.5 * predicted_heights


2. clipping boxes


refpath: ``fast_rcnn/bbox_transform.clip_box()``

::

    im_shape = (s1, s2)

    x1/x2:

        # compare x1 with (s2 - 1), get the minimum one
        if boxes[:, 0::4] < (s2 - 1): boxes[:, 0::4]
        else                        : (s2 - 1)
        
        # compare x1 with 0, get the maximum one
        if min(boxes[:, 0::4], (s2 - 1)) > 0: min(boxes[:, 0::4], (s2 - 1))
        else                                : 0

    y1/y2:

        # compare y1 with (s1 - 1), get the minimum one
        # compare new_y1 with 0, get the maximum one


3. removing predicted boxes with eighter height/width < threshold


::

    3.1. filter boxes by min_size

        min_size = min_size * im_info[2]

        widths' = widths - xs + 1
        heights' = heights - ys + 1

        keep = where(ws >= min_size) & (hs >= min_size)[0]


    3.2. get proposals and scores by keep_index

        proposals = proposals[keep, :]
        scores = scores[keep]


4. sorting all (proposals, scores) pairs by score from highest to lowest  
5. taking top pre_nms_topN

::

    - sorting scores from highest to lowest
      - sorting all scores from highest to lowest
      - if pre_nms_topN > 0: order = order[:pre_nms_topN]

         For instance: top 6000

    - proposals[order, :]
    - scores[order]   


6. applying nms  
7. taking after_nms_topN  
8. return the top proposals (RoIs top) 

::

    - applying nms

        - inputs: 
          - detections = np.hstack((proposals, scores))
          - nms
        - if detections.shape[0] == 0: return []
        - calculating keep:
            x1 = detections[:, 0]
            y1 = detections[:, 1]
            x2 = detections[:, 2]
            y2 = detections[:, 3]
            scores = detections[:, 4]

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.mamimum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]]) - inter)

                inds = np.where(ovr <= thresh)[0]
                order = order[inds + 1]

        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]

    - proposals[keep, :]
    - scores[keep]    


9. output rois blob
10. output scores blob

::

    rois blob:

        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))

    scores blob
    
        if len(top) > 1:
            top[1].reshape(*(scores.shape))
            top[1].data[...] = scores


Backward:



Source Codes
------------------------------


Test Examples
------------------------------
