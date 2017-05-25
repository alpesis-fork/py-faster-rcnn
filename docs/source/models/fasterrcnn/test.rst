Test
==============================================================================

Network Initialization
------------------------------------------------------------------------------


Image Preprocessing: (blobs) data
------------------------------------------------------------------------------

Image scale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key concepts
``````````````````````````````````````````````````````````````````````````````

- pixel mean
- image resize


Process Steps
``````````````````````````````````````````````````````````````````````````````

::

    1. image_copied = image (read from opecv)
    2. image_pixel_mean = image_copied - cfg.PIXEL_MEANS


    3. scale images:

        3.1. get image shape:
            image_shape = image_pixel_mean.shape
            image_size_min = min(shape)
            image_size_max = max(shape)

        3.2. scale an image

            3.2.1. image_scale
                image_scale = cfg.TEST.SCALES / image_size_min

            3.2.2. image_scale (MAX_SIZE)
                if (image_scale * image_size_max) > cfg.TEST.MAX_SIZE:
                    image_scale = cfg.TEST.MAX_SIZE / image_size_max

            3.2.3. image.resize
                image_resized = cv2.resize(image_scale)

        outputs:
         - image_scale_factors
         - resized_images


Images to blob
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process blob

refpath: ``lib/util/blob.im_list_to_blob``

::

    1. blob = image_list_to_blob(resized_images)

        blob: (num_images, max_shape[0], max_shape[1], 3)
        - initialize the blob: max_shape, num_images
        - fill the images into blob: image -> blob
        - blob transpose: blob.transpose(channel_swap) 
          - channel_swap = (batch elem, channel, height, width) = (0,3,1,2)


        
Image Preprocessing: (blobs) roi
------------------------------------------------------------------------------

refpath: ``lib/fast_rcnn/test._get_rois_blob()``

::

    

Image Detection
------------------------------------------------------------------------------

Process steps:

::

    1. process input data: blobs (data, im_info, rois)
    2. forward network: blobs_out
    3. process output data: scores, pred_boxes


Get blobs and rois

- blobs: input preprocessed images into blobs['data']
- rois: 
   - _get_rois_blob(rois, image_scale_factors)
   - filter duplicated rois

refpath: ``lib/fast_rcnn/test.im_detect()``

::

    blobs = {
        'data': <image_resized>,
        'rois': if cfg.TEST.HAS_RPN == False
        'im_info': if cfg.TEST.HAS_RPN == True
    }

    - data:
    - rois:
    - im_info: [blobs['data'].shape[2], blobs['data'].shape[3], image_scale[0]] 

    if cfg.TEST.HAS_RPN:
    - rois
    - boxes

    scores: 
    - if cfg.TEST.SVM: net.blobs['cls_score']
    - else:            blobs_out['cls_prob']
     

    outputs:
    - scores
    - pred_boxes


    1. data: _get_image_blob(image)
    2. rois: _get_rois_blob(rois, image_scale_factors) if not cfg.TEST.HAS_RPN
    
    3. check duplicated rois and boxes (TODO)

    4. im_info

        if cfg.TEST.HAS_RPN:
            blobs['im_info'] = [blobs['data'].shape[2],
                                blobs['data'].shape[3],
                                image_scale[0]]


    5. reshape network inputs: blobs[''].shape -> net.blobs[''].reshape()

        net.blobs['data'].reshape(*blobs['data'].shape))
    
        if cfg.TEST.HAS_RPN: net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
        else:                net.blobs['rois'].reshape(*(blobs['rois'].shape)) 

    6. network forward:

        blobs_out = net.forward(**forward_kwargs)
        
        forward_kwargs:
         - 'data': blobs['data']
         - 'im_info': if cfg.TEST.HAS_RPN == True
         - 'rois': if cfg.TEST.HAS_RPN == False 

        blobs_out:
           {
               'bbox_pred': [],
               'cls_prob': []

    7. unscale back to raw image space (if cfg.TEST.HAS_RPN)
        
        if cfg.TEST.HAS_RPN:
            rois = net.blobs['rois'].data.copy()
            boxes = rois[:, 1:5] / im_scales[0]


    8. get scores:

        if cfg.TEST.SVM: scores = net.blobs['cls_score'].data
        else:            scores = blobs_out['cls_prob']


    9. get pred_boxes:

        if cfg.TEST.BBOX_REG:
            box_deltas = blobs_out['bbox_pred']
            pred_boxes = bbox_transform_inv(boxes, box_deltas)
            pred_boxes = clip_boxes(pred_boxes, im.shape)
        else:
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    10. map scores and predictions back to the original set of boxes

        if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
            scores = scores[inv_index, :]
            pred_boxes = pred_boxes[inv_index, :]

Post Processing: bbox
------------------------------------------------------------------------------

refpath: ``lib/fast_rcnn/bbox_transform.bbox_transform_inv()``

refpath: ``lib/fast_rcnn/clip_boxes``

Visualization
------------------------------------------------------------------------------

- classes
- boxes
- scores
- nms
- viz

refpath: ``tools/demo.demo()``

refpath: ``lib/fast_rcnn/nms_wrapper.py``

refpath: ``lib/nms/cpu_nms.pyx``
   
