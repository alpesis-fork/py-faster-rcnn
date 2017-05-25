Softmax
==============================================================================


Key Concepts
------------------------------


Mathematical Equations
------------------------------

Process Steps
------------------------------

Forward:

``softmax for cls_score``

For each roi, calculating the probability of each class, thus the output shape
should be (300, 21).

- for each roi, find out the max value from 21 classes -> scale_data
- for each class per roi, calculate z[21, 1] = scale_data * sum_multiplier[21, 1] + z[21, 1] = scale_data + z[21, 1]
- for each class per roi, calculate e^z[21, 1]
- for all classes per roi, calculate sum(e^z[21, 1]) = top_data[300, 21] * sum_multiplier[21, 1] = scale_data[300, 1]
- for each class per roi, calculate e^z / sum(e^z)

Thus, cls_score -> cls_prob.

For example:

- scale_data: [300, 1] -> find the max score for each class from cls_score[300, 21]
- z [300, 21] = scale_data[300, 1] + top_data[300, 21], scale_data * each score per roi
- e^z
- sum(e^z) with all classes per score (gemv for sum, sum_multiplier[1,1,...,1])
- e^z / sum(e^z)

::

    for each roi (300):

      1. scale_data for each class (300)
        1.1. initialize the scale_data (300, 1)
        1.2. find out the max value from all classes per roi
             for channels:
             for inner_num_:
                 scale_data[k] = max(scale_data[k], bottom_data[i*dim + j * inner_num_ + k])

      2. subtraction

         top_data[21, 1] = -sum_multiplier_[21, 1] * scale_data[1, 1] + top_data[21, 1]

      3. exponentiation

         top_data[21, 1] = e^(top_data[21, 1])

      4. sum after exp

         scale_data = top_data[21, 1] * sum_multiplier_[21, 1]

      5. division

         top_data[i] = top_data/scale_data

Backward:

::

    1.


Source Codes
------------------------------

Forward: 

``layers/softmax_layer.cpp``

::

    template <typename Dtype>
    void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) 
    {
      std::cout << "(SoftmaxLayer) forward_cpu: " << std::endl;

      const Dtype* bottom_data = bottom[0]->cpu_data();
      Dtype* top_data = top[0]->mutable_cpu_data();
      Dtype* scale_data = scale_.mutable_cpu_data();

      int channels = bottom[0]->shape(softmax_axis_);
      int dim = bottom[0]->count() / outer_num_;
      caffe_copy(bottom[0]->count(), bottom_data, top_data);

      // We need to subtract the max to avoid numerical issues, compute the exp,
      // and then normalize.
      for (int i = 0; i < outer_num_; ++i) 
      {
        // initialize scale_data to the first plane
        caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
        for (int j = 0; j < channels; j++) 
        {
          for (int k = 0; k < inner_num_; k++) 
          {
            scale_data[k] = std::max(scale_data[k], bottom_data[i * dim + j * inner_num_ + k]);
          }
        }

        // subtraction
        caffe_cpu_gemm<Dtype>(CblasNoTrans, 
                              CblasNoTrans, 
                              channels, 
                              inner_num_,
                              1, 
                              -1., 
                              sum_multiplier_.cpu_data(), 
                              scale_data, 
                              1., 
                              top_data);

        // exponentiation
        caffe_exp<Dtype>(dim, top_data, top_data);

        // sum after exp
        caffe_cpu_gemv<Dtype>(CblasTrans, 
                              channels, 
                              inner_num_, 
                              1.,
                              top_data, 
                              sum_multiplier_.cpu_data(), 
                              0., 
                              scale_data);

        // division
        for (int j = 0; j < channels; j++) 
        {
          caffe_div(inner_num_, top_data, scale_data, top_data);
          top_data += inner_num_;
        }
      }
    }



Backward:

::


Test Examples
------------------------------
