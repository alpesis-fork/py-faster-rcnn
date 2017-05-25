Convolution
==============================================================================


Key Concepts
------------------------------


Mathematical Equations
------------------------------

Process Steps
------------------------------

conv_im2col_cpu()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params: data, col_buff

    2. im2col_cpu()

        if (!force_nd_im2col_ && num_spatial_axes_ == 2):
            im2col_cpu(data,
                       conv_in_channels_,
                       conv_iput_shape_.cpu_data()[1],
                       conv_input_shape_.cpu_data()[2],
                       kernel_shape_.cpu_data()[0],
                       kernel_shape_.cpu_data()[1],
                       pad_.cpu_data()[0],
                       pad_.cpu_data()[1],
                       stride_.cpu_data()[0],
                       stride_.cpu_data()[1],
                       dilation_.cpu_data()[0],
                       dilation_.cpu_data()[1],
                       col_buff)

        else:
            im2col_nd_cpu(data,
                          num_spatial_axes_,
                          conv_input_shape_.cpu_data(),
                          col_buffer_shape_.data(),
                          kernel_shape_.cpu_data(),
                          pad_.cpu_data(),
                          stride_.cpu_data(),
                          dilation_.cpu_data(),
                          col_buff)

conv_col2im_cpu()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params: col_buff, data

    2. col2im_cpu()

        if (!force_nd_im2col_ && num_spatial_axes_ == 2):
            col2im_cpu(col_buff,
                       conv_in_channels_,
                       conv_input_shape_.cpu_data()[1],
                       conv_input_shape_.cpu_data()[2],
                       kernel_shape_.cpu_data()[0],
                       kernel_shape_.cpu_data()[1],
                       pad_.cpu_data()[0],
                       pad_.cpu_data()[1],
                       stride_.cpu_data()[0],
                       stride_.cpu_data()[1],
                       dilation_.cpu_data()[0],
                       dilation_.cpu_data()[1],
                       data)

        else:
            col2im_nd_cpu(col_buff,
                          num_spatial_axes_,
                          conv_input_shape_.cpu_data(),
                          col_buffer_shape_.data(),
                          kernel_shape_.cpu_data(),
                          pad_.cpu_data(),
                          stride_.cpu_data(),
                          dilation_.cpu_data(),
                          data)


im2col_cpu()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params: data_im, channels, height, width, 
                     kernel_h, kernel_w, 
                     pad_h, pad_w,
                     stride_h, stride_w,
                     dilation_h, dilation_w
                     data_col

    2. output_h, output_w

        output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
        output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1

    3. channel_size = height * width

    4. im2col

        for (channel = channels; channel--; data_im += channel_size)
            for (kernel_row = 0; kernel_row < kernel_h; kernel_row++)
                for (kernel_col = 0; kernel_col < kernel_w; kernel_col++)
                
                    input_row = -pad_h + kernel_row * dilation_h

                    for (output_rows = output_h; output_rows; output_rows--)

                        if (!is_a_ge_zeros_and_a_lt_b(input_row, height))
                            for (output_cols = output_w; output_cols; output_cols--)
                                *(data_col++) = 0

                        else:

                            input_col = -pad_w + kernel_col * dilation_w

                            for (output_col = output_w; output_col; output_col--)

                                if (is_a_ge_zero_and_a_lt_b(input_col, width)
                                    *(data_col++) = data_im[input_row * width + input_col]

                                else:
                                    *(data_col++) = 0

                                input_col += stride_w

                        input_row += stride_h
    

im2col_nd_cpu()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params:


im2col_nd_core_cpu()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params:

col2im_cpu()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params:

col2im_nd_cpu()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params:

col2im_nd_core_cpu()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    1. input params:

is_a_ge_zero_and_a_lt_b()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input: a, b

    2. return: bool a < b


LayerSetUp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. read conv_param from caffe proto: ConvolutionParameter

        - num_output: 
        - bias_term: [default: true]
        - pad: [default: 0]
        - kernel_size: 
        - stride: [default: 1]
        - dilation: [default: 1]
        - pad_h: [default: 0]
        - pad_w: [default: 0]
        - kernel_h: 
        - kernel_w:
        - stride_h:
        - stride_w:
        - group: [default: 1]
        - weight_filler: 
        - bias_filler: 
        - engine: [default: 0]
        - axis: [default: 1]
        - force_nd_im2col: [default: false]

    2. set up the params:

        2.1. force_nd_imcol_ = conv_param.force_nd_im2col()

        2.2. channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis())

            - CHECK: -shape.size() < axis() < shape.size()
            - if axis() < 0: axis + shape.size()
            - return axis

        2.3. num_spatial_axes_

            num_spatial_axes_ = input->shape.size - first_spatial_axis
            first_spatial_axis = channel_axis_ + 1 (get the H/W index)

        2.4. bottom_dim_blob_shape(1, num_spatial_axes_ + 1)

        2.5. spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1))

        2.6. kernel_shape_

            kernel_shape_.Reshape(spatial_dim_blob_shape)
            int* kernel_shape_data = kernel_shape_.mutable_cpu_data()

            kernel_shape_data[0] = conv_param.kernel_h()
            kernel_shape_data[1] = conv_param.kernel_w()

            or
            
            kernel_shape_data[i] = conv_param.kernel_size(i)

            CHECK: kernel_shape_data[i] > 0

        2.7. stride_

            stride_.Reshape(spatial_dim_blob_shape);
            int* stride_data = stride_.mutable_cpu_data()

            stride_data[0] = conv_param.stride_h()
            stride_data[1] = conv_param.stride_w()
           
            or 

            stride_data[i] = conv_param.stride(i)

            CHECK: stride_data[i] > 0

        2.8. pad_

            pad_.Reshape(spatial_dim_blob_shape)
            int* pad_data = pad_.mutable_cpu_data()

            pad_data[0] = conv_param.pad_h()
            pad_data[1] = conv_param.pad_w()

            or

            pad_data[i] = conv_param.pad(i)


        2.9. dilation_

            dilation_.Reshape(spatial_dim_blob_shape)
            int* dilation_data = dilation_.mutable_cpu_data()

            dilation_data[i] = conv_param.dilation(i) 

        2.10. is_1x1_

            Special case: im2col is the identity for 1x1 convolution with stride 1
            and no padding, so flag for skipping the buffer and transformation

            is_1x1_ = true
            for (int i = 0; i < num_spatial_axes_; ++i)
                is_1x1_ &= kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0
                if (!is_1x1_): break

        2.11. channels_

            channels_ = bottom[0]->shape(channel_axis_)

        2.12. num_output_

            num_output_ = this->layer_param_.convolution_param().num_output()
            num_output_ > 0

        2.13. group_

            group_ = this->layer_param_.convolution_param().group()

            CHECK: channels_ % group_ == 0
            CEHCK: num_output_ % group_ == 0

        2.14. conv_in_channels_ & conv_out_channels_

            if (reverse_dimensions())
                conv_out_channels = channels_
                conv_in_channels = num_output_
            else
                conv_out_channels = num_output_
                conv_in_channels = channels_

        2.15. weights (blobs_[0]: filter weights)

            weight_shape(2):
            - weight_shape[0] = conv_out_channels_
            - weight_shape[1] = conv_in_channels_ / group_

            for (int i = 0; i < num_spatial_axes_; ++i)
                weight_shape.push_back(kernel_shape_data[i])

        2.16. bias_term_ (blobs_[1]: biases (optional))

            bias_term_ = this->layer_param_.convolution_param().bias_term()
            bias_shape(bias_term_, num_output_)

        2.17. this->blobs_: weights & biases

            if this->blobs_.size() > 0:

                CHECK: 1 + bias_term_ == this->blobs_.size()
                CHECK: weight_shape == this->blobs_[0]->shape()
                CHECK: bias_term_ && bias_shape == this->blobs_[1]->shape()

            else:

                if (bias_term_): this->blobs_.resize(2)
                else           : this->blobs_.resize(1)

                // initialize and fill the weights
                // shape = output_channels * input_channels_per_group * kernel_h * kernel_w
               this->blobs_[0].reset(new Blob<Dtype>(weight_shape))
               shared_ptr<Filler<Dtype>> weight_filler(GetFiller<Dtype>(this->weight_filler()))
               weight_filler->Fill(this->blobs_[0].get())

               // if necessary, initialize and fill the biases
               this->blobs_[1].reset(new Blob<Dtype>(bias_shape))
               shared_ptr<Filler<Dtype>> bias_filler(GetFiller<Dtype>(this->bias_filler()))
               bias_filler->Fill(this->blobs_[1].get())


        2.18. kernel_dim_

            kernel_dim_ = this->blobs_[0]->count(1)

        2.19. weight_offset_ 

            weight_offset_ = conv_out_channels_ * kernel_dim_ / group_

        2.20. this->param_propagate_down_

            // propagate gradients to the parameters (as directed by backward pass)
            this->param_propagate_down_.resize(this->blobs_.size(), true)


Reshape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. CHECK: bottom[0].shape_size = first_spatial_axis + num_spatial_axes

        first_spatial_axis = channel_axis_ + 1
        CHECK: bottom[0]->shape.size() == (first_spatial_axis + num_spatial_axes_)

    2. CHECK: bottom[0].channel_axis_ == channels_ 

        num_ = bottom[0]->count(0, channel_axis_)
        CHECK: bottom[0]->shape(channel_axis_) == channels_

    3. CHECK: bottom[0]->shape() == bottom[bottom_id]->shape()

        for 1 to bottom.size():
            CHECK: bottom[0]->shape() == bottom[bottom_id]->shape()

    4. tops_shape

        bottom_shape_ = &bottom[0]->shape()
        output_shape_ = compute_output_shape()
        top_shape(bottom[0]->shape.begin(), bottom[0]->shape().begin() + channel_axis_)
        top_shape.push_back(num_output_)

        for i = 0 to num_spatial_axes_:
            top_shape.push_back(output_shape_[i])


    4. top->reshape()

        for top_id = 0 to top.size():
            top[top_id]->Reshape(top_shape)


    5. conv_out_spatial_dim_

        if reverse_dimensions():
            conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis)
        else:
            conv_out_spatial_dim_ = top[0]->count(first_spatial_axis)

    6. col_offset_

        col_offset_ = kernel_dim_ * conv_out_spatial_dim_

    7. output_offset_

        output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_

    8. conv_input_shape_

        vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1)
        conv_input_shape_.Reshape(bottom_dim_blob_shape)

        conv_input_shape_data = conv_input_shape_.mutable_cpu_data()
        for i = 0 to num_spatial_axes_ + 1:
            if reverse_dimensions(): conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i)
            else                   : conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i)

    9. col_buffer_shape_

        // im2col: only hold one image at a time
        //         if 1x1 convolution, unused to save memory
        col_buffer_shape_.clear()
        col_buffer_shape_.push_back(kernel_dim_ * group_)

        for i = 0 to num_spatial_axes_:
            if reverse_dimensions(): col_buffer_shape_.push_back(input_shape(i+1))
            else                   : col_buffer_shape_.push_back(output_shape_[i])

    10. col_buffer_

        col_buffer_.Reshape(col_buffer_shape_)

    11. bottom_dim_, top_dim_

        bottom_dim_ = bottom[0]->count(channel_axis_)
        top_dim_ = top[0]->count(channel_axis_)

    12. num_kernels_im2col_, num_kernels_col2im_

        num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_
        num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_


    13. out_spatial_dim_

        out_spatial_dim_ = top[0]->count(first_spatial_axis)

    14. bias_multiplier_

        if (bias_term_):
            bias_multiplier_shape(1, out_spatial_dim_)
            bias_multiplier_.Reshape(bias_multiplier_shape)
            caffe_set(bias_multiplier_.count(), Dtype(1), bias_multiplier_.mutable_cpu_data())

compute_output_shape()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. get kernel_shape_data, stride_data, pad_data, dilation_data

        kernel_shape_data = this->kernel_shape_.cpu_data()
        stride_data = this->stride_.cpu_data()
        pad_data = this->pad_.cpu_data()
        dilation_data = this->dilation_.cpu_data()
        this->output_shape_.clear()

    2. output_shape_

        for i = 0 to num_spatial_axes_:
            input_dim = this->input_shape(i+1)
            kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1
            output_dim = (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1
            this->output_shape_.push_back(output_dim)


forward_cpu_gemm()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params: input, weights, output, skip_im2col

    2. im2col -> col_buff

        col_buff = input
        
        if (!is_1x1_)
            if !skip_im2col
                conv_im2col_cpu(input, col_buffer_.mutable_cpu_data())
            col_buff = col_buffer_.cpu_data()

    3. caffe_cpu_gemm()

        for group = 0 to group_:
            caffe_cpu_gemm(no, no, conv_out_channels_ / group_, 
                                   conv_out_spatial_dim_,
                                   kernel_dim_,
                                   1,
                                   weights + weight_offset_ * g,
                                   col_buff + col_offset_ * g,
                                   0,
                                   output + output_offset_ * g)


forward_cpu_bias()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params: output, bias

    2. caffe_cpu_gemm()

        caffe_cpu_gemm(no, no, num_output_,
                               out_spatial_dim_,
                               1,
                               1,
                               bias,
                               bias_multiplier_.cpu_data(),
                               1,
                               output)


backward_cpu_gemm()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params: output, weights, input

    2. col_buff

        col_buff = col_buffer_.mutable_cpu_data()
        if (is_1x1_): col_buff = input

        for group = 0 to group_:
            caffe_cpu_gemm(yes, no, kernel_dim_,
                                    conv_out_spatial_dim_,
                                    conv_out_channels_ / group_,
                                    1,
                                    weights + weight_offset_ * g,
                                    output + output_offset_ * g,
                                    0,
                                    col_buff + col_offset_ * g)

    3. conv_col2im_cpu(col_buff, input)

        if (!is_1x1_):
            conv_col2im_cpu(col_buff, input)

weight_cpu_gemm()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params: input, output, weights

    2. col_buff

        col_buff = input
    
        if (!is_1x1_):
            conv_im2col_cpu(input, col_buffer_.mutable_cpu_data())
            col_buff = col_buffer_.cpu_data()

    3. caffe_cpu_gemm()

        caffe_cpu_gemm(no, yes, conv_out_channels_ / group_,
                                kernel_dim_,
                                conv_out_spatial_dim_,
                                1,
                                output + output_offset_ * g,
                                col_buff + col_offset_ * g,
                                1,
                                weights + weight_offset_ * g)

backward_cpu_bias()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params: bias, input

    2. caffe_cpu_gemv()

        caffe_cpu_gemv(no, num_output_,
                           out_spatial_dim_,
                           1,
                           input,
                           bias_multiplier_.cpu_data(),
                           1,
                           bias)


Forward_cpu()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params: bottom, top

    2. weight

        weight = this->blobs_[0]->cpu_data()

    3. forward

        for i = 0 to bottom.size():

            bottom_data = bottom[i]->cpu_data()
            top_data = top[i]->mutable_cpu_data()

            for n = 0 to num_:
                forward_cpu_gemm(bottom_data + n * bottom_dim_,
                                 weight,
                                 top_data + n * this->top_dim_)

            if (bias_term_):
                bias = blobs_[1]->cpu_data()
                forward_cpu_bias(top_data + n * top_dim_, bias)


Backward_cpu()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1. input params: top, propagate_down, bottom

    2. weight, weight_diff

        weight = this->blobs_[0]->cpu_data()
        weight_diff = this->blobs_[0]->mutable_cpu_diff()

    3. backward

        for i = 0 top.size():
            top_diff = top[i]->cpu_diff()
            bottom_data = bottom[i]->cpu_data()
            bottom_diff = bottom[i]->mutable_cpu_diff()
    
            // bias graident, if necessary
            if (bias_term_ && param_propagate_down_[1]):
                bias_diff = blobs_[1]->mutable_cpu_diff()
       
                for n = 0 to num_:
                    backward_cpu_bias(bias_diff, top_diff + n * top_dim_)

            if (param_propagate_down_[0] || propagate_down[i])

                for n = 0 to num_:
                    if param_propagate_down_[0]:
                        weight_cpu_gemm(bottom_data + n * bottom_dim_,
                                        top_diff + n * top_dim_,
                                        weight_diff)

                    if (propagate_down[i]):
                        backward_cpu_gemm(top_diff + n * top_dim_,
                                          weight,
                                          bottom_diff + n * bottom_dim_)

Forward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- image: (800, 600)
- padding: (1, 1)
- channels: (R,G,B)

- kernel: (3, 3)
- stride: (1, 1)
- dilation: (1, 1)

Calculations:

- # of outputs = (W - F + 2P)/S + 1
- spatial_size: [h, w]
- padding_size: [h, w]
- kernel_size: [h, w]
- stride_size: [h, w]
- n_outputs_w = (spatial_w - kernel_w + 2 * padding_w) / stride_w + 1
- n_outputs_h = (spatial_h - kernel_h + 2 * padding_h) / stride_h + 1

# of parameters:

- image: [3, 32, 32]
- kernel: [5, 5]
- stride: [1, 1]
- padding: [2, 2]
- # of kernels: 10
- # of params = ((kernel_h * kernel_w * channels) + 1) * # of kernels = ((5*5*3)+1)*10 = 760 
- 1 for bias

For example:

# of feature maps:

- kernel size: (32, 16, 16)
- image: (256, 256)
- # of feature maps: 32 * (image_size - kernel_size + 1) = 32 * (256 - 16 + 1) = (32, 241, 241)

# of feature maps:

- spatial_size: [800, 600]
- kernel_size: [3, 3]
- padding_size: [1, 1]
- stride_size: [1, 1]
- n_outputs_w = (800 - 3 + 2 * 1) / 1 + 1 = 800
- n_outputs_h = (600 - 3 + 2 * 1) / 1 + 1 = 600

::

    1. initialization

        weight =
        
        for bottom.size
            bottom_data =
            top_data = 

            for (int n = 0; n < this->num_; ++n)
                forward_cpu_gemm(bottom_data + n * this->bottom_dim_,
                                 weight,
                                 top_data + n * this->top_dim_)

            if (this->bias_term_)
                bias = this->blobs_[1]->cpu_data()
                this->forward_cpu_bias(top_data + n * this->top_dim_, bias)


    2. forward_cpu_gemm


       2.1. im2col

        col_buff = input

        if (!is_1x1_)
            if (!skip_im2col)
                conv_im2col_cpu(input, col_buffer_)
            col_buff = col_buffer_

            # conv_im2col_cpu
            if (!force_nd_im2col_ && num_spatial_axes_ == 2):
                im2col_cpu()
            else:
                im2col_nd_cpu()

            2.1.1. im2col_cpu()

                output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1
                output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1
        
                channel_size = height * width
                for channels:
                for kernel_row:
                for kernel_col:
                    input_row = -pad_h + kernel_row * dilation_h
                    for output_rows:
                       if (!is_a_ge_zero_and_a_lt_b(input_row, height))
                         for output_cols:
                           *(data_col++) = 0
                       else:
                         input_col = -pad_w + kernel_col * dilation_w
                         for output_cols:
                           

            2.1.2. im2col_nd_cpu()
                kIm2Col = true
                im2col_nd_core_cpu()

      2.2. conv

        for group_: (g)

            C: = (weights + weight_offset_ * g ) * (col_buff + col_offset_ * g)
            - weights: [conv_out_channels_ / group_, conv_out_spatial_dim_]
            - X: [conv_out_spatial_dim, kernel_dim_]

            gemm():
            - M: conv_out_channels_ / group_
            - N: conv_out_spatial_dim_
            - K: kernel_dim_
            - alpha: 1
            - A: weights + weight_offset_ * g
            - B: col_buff + col_offset_ * g
            - beta: 0
            - C: output + output_offset_ * g


    3. forward_cpu_bias

        output = bias * bias_multiplier_ + output
        - bias: [num_output_, out_spatial_dim_]
        - bias_multiplier: [out_spatial_dim, 1]

        gemm():
        - M: num_output_
        - N: out_spatial_dim_
        - K: 1
        - alpha: 1
        - A: bias
        - B: bias_multiplier_
        - beta: 1
        - C: output


Backward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Source Codes
------------------------------


Test Examples
------------------------------
