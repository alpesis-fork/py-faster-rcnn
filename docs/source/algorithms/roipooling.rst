ROIPooling
==============================================================================


Key Concepts
------------------------------


Mathematical Equations
------------------------------

Process Steps
------------------------------

Forward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rois on original image -> rois on feature map -> roi on pooled feature map

1. rois on original image -> rois on feature map: [x, y, w, h] * spatial scale  

2. rois on feature map: (w, h)  

3. rois on feature map -> rois on pooled feature map

calculating the spatial scale

- rois heights / pooled height
- rois widths / pooled width

4. get the roi corresponding feature map data

5. for each channel, (input) feature map -> (output) pooled feature map

- roi on pooled feature map size -> roi on feature map size: [wstart, hstart, wend, hend]
- get [wstart, hstart, wend, hend] on roi on feature map
- check area
- output pooled feature map

refpath: ``src/caffe/layers/roi_pooling_layers.cpp``

::

    1. get bottom / top data

       bottom:
         bottom_data = bottom[0]
         batch_size = bottom[0]->num()
         bottom_rois = bottom[1]
         n_rois = bottom[1]->num()

       top:
         top_data = top[0]
         top_count = top[0]->count()


    2. assign the size of top_data and argmax_data and initialize each value

        caffe_set(top_count, Dtype(-FLT_MAX), top_data)
          - assign the memory (size) of (-FLT_MAX) to each value of top_data
          - assign the value -FLT_MAX to each value of top_data

        caffe_set(top_count, -1, argmax_data)
          - assign the memory (size) of -1
          - assign -1 to argmax_data


    3. max pooling of ROI (x1, y1, x2, y2)


        3.1. roi axis on original image -> feature map axis

          bottom_roi: [roi_batch_ind, roi_start_w, roi_start_h, roi_end_w, roi_end_h]

          - roi_batch_ind = bottom_rois[0]
          - roi_start_w = round(bottom_rois[1] * spatial_scale_)
          - roi_start_h = round(bottom_rois[2] * spatial_scale_)
          - roi_end_w = round(bottom_rois[3] * spatial_scale_)
          - roi_end_h = round(bottom_rois[4] * spatial_scale_)

        3.2. check roi_batch_ind
            
          - roi_batch_ind == 0
          - roi_batch_ind < batch_size

        3.3. calculate each roi size (roi_height, roi_width) on feature map

          # calculate roi size on feature map
          - roi_height = max(roi_end_h - roi_start_h + 1, 1)
          - roi_width = max(roi_end_w - roi_start_w + 1, 1)

          # pooled_feature_map <- pooling feature map
          - bin_size_h = roi_height / pooled_height_
          - bin_size_w = roi_width / pooled_width_

          - batch_data = bottom_data + bottom[0]->offset(roi-batch_ind)

          - roi pooling: channels, pooled_height_, pooled_width_

              # output size <- input size
              - hstart = floor(pooled_height_ * bin_size_h)
              - wstart = floor(pooled_width_ * bin_size_w)
              - hend = ceil((pooled_height_ + 1) * bin_size_h)
              - wend = ceil((pooled_width_ + 1) * bin_size_w)
               
              - hstart = min(max(hstart + roi_start_h, 0), height_)
              - hstart = min(max(hend + roi_start_h, 0), height_)
              - wstart = min(max(wstart + roi_start_w, 0), width_)
              - wend = min(max(wend + roi_start_w, 0), width_)

              - bool is_empty = (hend <= hstart) || (wend <= wstart)

              # output index
              - const int pool_index = ph * pooled_width_ + pw

              - if (is_empty)
                - top_data[pool_index] = 0
                - argmax_data[pool_index] = -1

              - hstart, hend, wstart, wend
                - index = h * width_ + w
                - if (bach_data[index] > top_data[pool_index])
                  - top_data[pool_index] = batch_data[index]
                  - argmax_data[pool_index] = index                   


        3.4. increment all data pointers by one channel

            batch_data += bottom[0]->offset(0, 1)
            top_data += top[0]->offset(0, 1)
            argmax_data += max_idx_.offset(0, 1)


        3.5. increment ROI data pointer

            bottom_rois += bottom[1]->offset(1)



Backward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Source Codes
------------------------------


Test Examples
------------------------------
