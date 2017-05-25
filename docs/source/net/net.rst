##############################################################################
Net
##############################################################################


::

    1. load the network (prototxt)

    2. parse the network
        2.1. filter layers: included / excluded 
        2.2. parse layer parameters: name, shape, params  
        2.3. initialize the variables (memory/size) corresponding to the layer parameters  

    3. intialize the network
        3.1. initialize the blob name
        3.2. initialize the layer

    4. load the weights (caffemodel)

    5. (test) image detection

        5.1. load the test image

        5.2. image -> blob
            5.2.1. image -> blob
            5.2.2. calculate the pixel mean
            5.2.3. start the image preprocessing: scale and resize the image
            5.2.4. put the resized image into the blob
            5.2.5. blob transposed (blob['data'])
            5.2.6. process blob['rois']
            5.2.7. process blob['im_info']
            5.2.8. reshape the blob['data'], blob['im_info'] 

        5.3. network forward

            layer: 0, (conv1_1) conv: reshape -> forward
            layer: 1, (relu1_1) relu:
            layer: 2, (conv1_2) conv: reshape -> forward
            layer: 3, (relu1_2) relu:
            layer: 4, (pool1) pooling:
            layer: 5, (conv2_1) conv: reshape -> forward
            layer: 6, (relu2_1) relu:
            layer: 7, (conv2_2) conv: reshape -> forward
            layer: 8, (relu2_2) relu:
            layer: 9, (pool2) pooling:
            layer: 10, (conv3_1) conv: reshape -> forward
            layer: 11, (relu3_1) relu:
            layer: 12, (conv3_2) conv: reshape -> forward
            layer: 13, (relu3_2) relu:
            layer: 14, (conv3_3) conv: reshape -> forward
            layer: 15, (relu3_3) relu:
            layer: 16, (pool3) pooling:
            layer: 17, (conv4_1) conv: reshape -> forward
            layer: 18, (relu4_1) relu:
            layer: 19, (conv4_2) conv: reshape -> forward
            layer: 20, (relu4_2) relu:
            layer: 21, (conv4_3) conv: reshape -> forward
            layer: 22, (relu4_3) relu:
            layer: 23, (pool4) pooling:
            layer: 24, (conv5_1) conv: reshape -> forward
            layer: 25, (relu5_1) relu:
            layer: 26, (conv5_2) conv: reshape -> forward
            layer: 27, (relu5_2) relu:
            layer: 28, (conv5_3) conv: reshape -> forward
            layer: 29, (relu5_3) relu:

            layer: 30, (conv5_3_relu5_3_0_split): 
            layer: 31, (rpn_conv/3x3) conv: reshape -> forward
            layer: 32, (rpn_relu/3x3) relu:
            layer: 33, (rpn/output_rpn_relu/3x3_0_split)
            layer: 34, (rpn_cls_score) conv: reshape -> forward
            layer: 35, (rpn_bbox_pred) conv: reshape -> forward
            layer: 36, (rpn_cls_score_reshape) reshape:

            layer: 37, (rpn_cls_prob) softmax: reshape -> forward
            layer: 38, (rpn_cls_prob_reshape) reshape:

            layer: 39, (proposal) proposal:

            layer: 40, (roi_pool5) roipooling:
            layer: 41, (fc6) innerproduct: reshape -> forward
            layer: 42, (relu6) relu:
            layer: 43, (fc7) innerproduct: reshape -> forward
            layer: 44, (relu7) relu:
            layer: 45, (fc7_relu7_0_split)
            layer: 46, (cls_score) innerproduct: reshape -> forward
            layer: 47, (bbox_pred) innerproduct: reshape -> forward
            layer: 48, (cls_prob) softmax: reshape -> forward


        5.4. blobs out: bbox_pred, cls_score

    6. net.blobs: rois, bbox_pred, cls_score

        6.1. get rois
        6.2. get bboxes: boxes * scale -> original image
        6.3. get scores:

    7. visualization:

        7.1. bbox_pred -> bbox_transform_inv() -> clip_boxes() -> bboxes
        7.2. process image_shape
        7.3. apply nms
        7.4. draw box and score
