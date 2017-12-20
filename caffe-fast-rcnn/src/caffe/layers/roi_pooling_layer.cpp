// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
// -----------------------------------------------------------------------------------------------

template <typename Dtype>
void ROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) 
{
  ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
  CHECK_GT(roi_pool_param.pooled_h(), 0) << "pooled_h must be > 0";
  CHECK_GT(roi_pool_param.pooled_w(), 0) << "pooled_w must be > 0";
  pooled_height_ = roi_pool_param.pooled_h();
  pooled_width_ = roi_pool_param.pooled_w();
  spatial_scale_ = roi_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}


template <typename Dtype>
void ROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) 
{

  // feature map: channels, height, width
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  // outputs: (N, C, H, W)
  // - N: # of RoIs
  // - C: channels
  // - H: pooled_height
  // - W: pooled_width
  top[0]->Reshape(bottom[1]->num(), 
                  channels_, 
                  pooled_height_,
                  pooled_width_);

  max_idx_.Reshape(bottom[1]->num(), 
                   channels_, 
                   pooled_height_,
                   pooled_width_);
}


template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) 
{
  std::cout << "(ROIPoolingLayer) Forward_cpu: " << std::endl;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  std::cout << "(ROIPoolingLayer) bottom_data.count(): " << bottom[0]->count() << std::endl;
  std::cout << "(ROIPoolingLayer) bottom_data.num(): " << bottom[0]->num() << std::endl;
  // std::cout << "(ROIPoolingLayer) bottom_data: " << std::endl;
  // for (int i = 0; i < bottom[0]->count(); ++i)
  //     std::cout << bottom_data[i] << ", ";
  // std::cout << std::endl;

  std::cout << "(ROIPoolingLayer) bottom_rois.count(): " << bottom[1]->count() << std::endl;
  std::cout << "(ROIPoolingLayer) bottom_rois.num(): " << bottom[1]->num() << std::endl;
  // std::cout << "(ROIPoolingLayer) bottom_rois: " << std::endl;
  // for (int i = 0; i < bottom[1]->count(); ++i)
  //     std::cout << bottom_rois[i] << ", ";
  // std::cout << std::endl;

  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  std::cout << "(ROIPoolingLayer) num_rois: " << num_rois << std::endl;
  std::cout << "(ROIPoolingLayer) batch_size: " << batch_size << std::endl; 


  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  std::cout << "(ROIPoolingLayer) top_data.count(): " << top_count << std::endl;
  std::cout << "(ROIPoolingLayer) top_data.num(): " << top[0]->num() << std::endl;
  std::cout << "(ROIPoolingLayer) top_data: " << std::endl;
  // for (int i = 0; i < top_count; ++i)
  //     std::cout << top_data[i] << ", ";
  // std::cout << std::endl;  

  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);


  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  std::cout << "(ROIPoolingLayer) spatial_scale_: " << spatial_scale_ << std::endl;
  std::cout << "(ROIPoolingLayer) pooled_height_: " << pooled_height_ << std::endl;
  std::cout << "(ROIPoolingLayer) pooled_width_: " << pooled_width_ << std::endl;
  for (int n = 0; n < num_rois; ++n) 
  {
    std::cout << "(ROIPoolingLayer) nth_rois: " << n << std::endl;

    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale_);
    int roi_start_h = round(bottom_rois[2] * spatial_scale_);
    int roi_end_w = round(bottom_rois[3] * spatial_scale_);
    int roi_end_h = round(bottom_rois[4] * spatial_scale_);
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    std::cout << "(ROIPoolingLayer) bottom_rois[0]: " << bottom_rois[0] << std::endl;
    std::cout << "(ROIPoolingLayer) roi_batch_ind: " << roi_batch_ind << std::endl;
    std::cout << "(ROIPoolingLayer) spatial_scale_: " << spatial_scale_ << std::endl;
    std::cout << "(ROIPoolingLayer) bottom_rois[1]: " << bottom_rois[1] << std::endl;
    std::cout << "(ROIPoolingLayer) roi_start_w: " << roi_start_w << std::endl;
    std::cout << "(ROIPoolingLayer) bottom_rois[2]: " << bottom_rois[2] << std::endl;
    std::cout << "(ROIPoolingLayer) roi_start_h: " << roi_start_h << std::endl;
    std::cout << "(ROIPoolingLayer) bottom_rois[3]: " << bottom_rois[3] << std::endl;
    std::cout << "(ROIPoolingLayer) roi_end_w: " << roi_end_w << std::endl;
    std::cout << "(ROIPoolingLayer) bottom_rois[4]: " << bottom_rois[4] << std::endl;
    std::cout << "(ROIPoolingLayer) roi_end_h: " << roi_end_h << std::endl;

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    std::cout << "(ROIPoolingLayer) roi_heihgt: " << roi_height << std::endl;
    std::cout << "(ROIPoolingLayer) roi_width: " << roi_width << std::endl;

    const Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width_);
    std::cout << "(ROIPoolingLayer) bin_size_h: " << bin_size_h << std::endl;
    std::cout << "(ROIPoolingLayer) bin_size_w: " << bin_size_w << std::endl;


    std::cout << "(ROIPoolingLayer) roi_batch_ind: " << roi_batch_ind << std::endl;
    std::cout << "(ROIPoolingLayer) bottom[0]->offset(roi_batch_ind): " << bottom[0]->offset(roi_batch_ind) << std::endl;
    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);
    std::cout << "(ROIPoolingLayer) batch_data: " << std::endl;
    for (int i = 0; i < bottom[0]->count(); ++i)
        std::cout << batch_data[i] << ", ";
    std::cout << std::endl;


    std::cout << "(ROIPoolingLayer) channels_: " << channels_ << std::endl;
    std::cout << "(ROIPoolingLayer) pooled_height_: " << pooled_height_ << std::endl;
    std::cout << "(ROIPoolingLayer) pooled_width_: " << pooled_width_ << std::endl;
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));

          // std::cout << "(ROIPoolingLayer) hstart: " << hstart << std::endl;
          // std::cout << "(ROIPoolingLayer) wstart: " << wstart << std::endl;
          // std::cout << "(ROIPoolingLayer) hend: " << hend << std::endl;
          // std::cout << "(ROIPoolingLayer) wend: " << wend << std::endl;

          hstart = min(max(hstart + roi_start_h, 0), height_);
          hend = min(max(hend + roi_start_h, 0), height_);
          wstart = min(max(wstart + roi_start_w, 0), width_);
          wend = min(max(wend + roi_start_w, 0), width_);
          // std::cout << "(ROIPoolingLayer) height_: " << height_ << std::endl;
          // std::cout << "(ROIPoolingLayer) width_: " << width_ << std::endl;
          // std::cout << "(ROIPoolingLayer) hstart (roi): " << hstart << std::endl;
          // std::cout << "(ROIPoolingLayer) hend (roi): " << hend << std::endl;
          // std::cout << "(ROIPoolingLayer) wstart (roi): " << wstart << std::endl;
          // std::cout << "(ROIPoolingLayer) wend (roi): " << wend << std::endl;

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          std::cout << "(ROIPoolingLayer) pool_index: " << pool_index << std::endl;
          if (is_empty) 
          {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }

          for (int h = hstart; h < hend; ++h) 
          {
            for (int w = wstart; w < wend; ++w) 
            {
              const int index = h * width_ + w;
              // std::cout << "(ROIPoolingLayer) h: " << h << std::endl;
              // std::cout << "(ROIPoolingLayer) width_: " << width_ << std::endl;
              // std::cout << "(ROIPoolingLayer) w: " << w << std::endl;
              // std::cout << "(ROIPoolingLayer) index: " << index << std::endl;
              if (batch_data[index] > top_data[pool_index]) 
              {
                // std::cout << "(ROIPoolingLayer) batch_data[index]: " << batch_data[index] << std::endl;
                // std::cout << "(ROIPoolingLayer) top_data[pool_index]: " << top_data[pool_index] << std::endl;
                top_data[pool_index] = batch_data[index];
                argmax_data[pool_index] = index;
              }
            }
          }
        }
      }

      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += max_idx_.offset(0, 1);
    }

    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}


template <typename Dtype>
void ROIPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROIPoolingLayer);
#endif

INSTANTIATE_CLASS(ROIPoolingLayer);
REGISTER_LAYER_CLASS(ROIPooling);

// -----------------------------------------------------------------------------------------------
}  // namespace caffe

