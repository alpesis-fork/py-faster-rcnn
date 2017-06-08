Pooling
==============================================================================


Key Concepts
------------------------------


Mathematical Equations
------------------------------

Process Steps
------------------------------

Forward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

max pooling

- top_mask:
- mask:

::

    1. initialization

        bool use_top_mask = top.size() > 1
        if use_top_mask
            set top_mask
        else
            set mask

        initialize top_data with value -FLT_MAX 

    2. max pooling

        for bottom[0].num()
        for channels
        for pooled_height_ (ph)
        for pooled_width_ (pw)

            hstart = ph * stride_h_ - pad_h_
            wstart = pw * stride_w_ - pad_w_
            hend = min(hstart + kernel_h_, height_)
            wend = min(wstart + kernel_w_, width_)
            hstart = max(hstart, 0)
            wstart = max(wstart, 0)

            pool_index = ph * pooled_width_ + pw

            for hstart -> hend (h)
            for wstart -> wend (w)
                index = h * width_ + w

                if (bottom_data[index] > top_data[pool_index])
                    top_data[pool_index] = bottom_data[index]
         
                    if use_top_mask
                        top_mask[pool_index] = index
                    else
                        mask[pool_index] = index                    
            

    3. offset

        bottom_data += bottom[0]->offset(0, 1)
        top_data += top[0]->offset(0, 1)
        if (use_top_mask)
            top_mask += top[0]->offset(0, 1)
        else
            mask += top[0]->offset(0, 1)

Source Codes
------------------------------


Test Examples
------------------------------
