InnerProduct
==============================================================================


Key Concepts
------------------------------

Matrix multiplication

GEMM and GEMV:

- GEMM(): computes the matrix-vector product for a general matrix or its transpose
- GEMV(): performs combined matrix multiplication and addition for general matrices or their transponses

Types:

- sgemm(): single precision, 32 bits floating point 
- dgemm(): double precision, 64 bits floating point
- sgemv(): single precision, 32 bits floating point
- dgemv(): double precision, 64 bits floating point


Mathematical Equations
------------------------------

SGEMM & DGEMM
~~~~~~~~~~~~~~

::

    C := alpha * A * B + beta * C

    - dim(A): m * n
    - dim(B): n * k
    - dim(C): m * k
    - alpha: scalar
    - beta: scalar

SGEMV & DGEMV
~~~~~~~~~~~~~~

::

    C := alpha * A  * x + beta * y 
    C := alpha * A' * x + beta * y

    - dim(A): m * n
    - x: float / double
    - y: float / double
    - alpha: float / double
    - beta: float / double


Process Steps
------------------------------

Forward
~~~~~~~~~~~~~~

::

    1. without bias: 

        C := (Dtype) 1. * bottom_data * weight + (Dtype) 0. * top_data

        - dim(bottom_data): M_
        - dim(weight): N_
        - dim(top_data): K_
        - alpha: (Dtype) 1.
        - beta: (Dtype) 0.

    2. with bias:

        C := (Dtype) 1. * bias_multiplier_.cpu_data() * this->blobs_[1]->cpu_data() + (Dtype) 1. * top_data

        - dim(bias_multiplier_.cpu_data()): M_
        - dim(this->blobs_[1]->cpu_data()): N_
        - dim(top_data): 1
        - alpha: (Dtype) 1.
        - beta: (Dtype) 1. 


Backward
~~~~~~~~~~~~~~

::

    1. this->param_propagate_down_[0]: caffe_cpu_gemm()

        C := (Dtype) 1. * top_diff * bottom_data + (Dtype) 1. * this->blobs_[0]->mutable_cpu_diff());
        
        - dim(top_diff): N_ * K_
        - dim(bottom_data): N_ * M_
        - dim(this->blobs_[0]->mutable_cpu_diff()): N_ * M_
        - alpha: (Dtype) 1.
        - beta: (Dtype) 1.


    2. bias_term_ && this->param_propagate_down_[1]: caffe_cpu_gemv()

        C := (Dtype) 1. * top_diff * bias_multiplier_.cpu_data() + (Dtype) 1. * this->blobs_[1]->mutable_cpu_diff()

        - dim(top_diff): M_ * N_
        - x: bias_multiplier_.cpu_data()
        - y: this->blobs_[1]->mutable_cpu_diff()
        - alpha: (Dtype) 1.
        - beta: (Dtype) 1.


    3. propagate_down[0]: caffe_cpu_gemm

        C := (Dtype) 1. * top_diff * this->blobs_[0]->cpu_data() + (Dtype) 0. * bottom[0]->mutable_cpu_diff()

        - dim(top_diff): M_ * K_
        - dim(this->blobs_[0]->cpu_data()): K_ * N_
        - dim(bottom[0]->mutable_cpu_diff()): M_ * N_
        - alpha: (Dtype) 1.
        - beta: (Dtype) 0.


Source Codes
------------------------------

Forward
~~~~~~~~~~~~~~~

refpath: ``src/caffe/layers/inner_product_layer.cpp``

::

	template <typename Dtype>
	void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
						   const vector<Blob<Dtype>*>& top) 
	{
	  std::cout << "(InnerProductLayer) forward_cpu: " << std::endl;

	  const Dtype* bottom_data = bottom[0]->cpu_data();
	  Dtype* top_data = top[0]->mutable_cpu_data();
	  const Dtype* weight = this->blobs_[0]->cpu_data();


	  caffe_cpu_gemm<Dtype>(CblasNoTrans, 
				CblasTrans, 
				M_, 
				N_, 
				K_, 
				(Dtype)1.,
				bottom_data, 
				weight, 
				(Dtype)0., 
				top_data);

	  if (bias_term_) 
	  {
	    caffe_cpu_gemm<Dtype>(CblasNoTrans, 
				  CblasNoTrans, 
				  M_, 
				  N_, 
				  1, 
				  (Dtype)1.,
				  bias_multiplier_.cpu_data(),
				  this->blobs_[1]->cpu_data(), 
				  (Dtype)1., 
				  top_data);
	  }
	}


Backward
~~~~~~~~~~~~~~

refpath: ``src/caffe/layers/inner_product_layer.cpp``


::


	template <typename Dtype>
	void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
						    const vector<bool>& propagate_down,
						    const vector<Blob<Dtype>*>& bottom) 
	{
	  if (this->param_propagate_down_[0]) 
	  {
	    const Dtype* top_diff = top[0]->cpu_diff();
	    const Dtype* bottom_data = bottom[0]->cpu_data();

	    // Gradient with respect to weight
	    caffe_cpu_gemm<Dtype>(CblasTrans, 
				  CblasNoTrans, 
				  N_, 
				  K_, 
				  M_, 
				  (Dtype)1.,
				  top_diff, 
				  bottom_data, 
				  (Dtype)1., 
				  this->blobs_[0]->mutable_cpu_diff());
	  }

	  if (bias_term_ && this->param_propagate_down_[1]) 
	  {
	    const Dtype* top_diff = top[0]->cpu_diff();
	    // Gradient with respect to bias
	    caffe_cpu_gemv<Dtype>(CblasTrans, 
				  M_, 
				  N_, 
				  (Dtype)1., 
				  top_diff,
				  bias_multiplier_.cpu_data(), 
				  (Dtype)1.,
				  this->blobs_[1]->mutable_cpu_diff());
	  }

	  if (propagate_down[0]) 
	  {
	    const Dtype* top_diff = top[0]->cpu_diff();
	    // Gradient with respect to bottom data
	    caffe_cpu_gemm<Dtype>(CblasNoTrans, 
				  CblasNoTrans, 
				  M_, 
				  K_, 
				  N_, 
				  (Dtype)1.,
				  top_diff, 
				  this->blobs_[0]->cpu_data(), 
				  (Dtype)0.,
				  bottom[0]->mutable_cpu_diff());
	  }
	}


GEMM
~~~~~~~~~~~~~~


refpath: ``src/caffe/util/math_functions.cpp``

::

	template<>
	void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
				   const CBLAS_TRANSPOSE TransB, 
				   const int M, 
				   const int N, 
				   const int K,
				   const float alpha, 
				   const float* A, 
				   const float* B, 
				   const float beta,
				   float* C) 
	{
	  std::cout << "(util::math_functions) caffe_cpu_gemm: " << std::endl;

	  int lda = (TransA == CblasNoTrans) ? K : M;
	  int ldb = (TransB == CblasNoTrans) ? N : K;
	  cblas_sgemm(CblasRowMajor, 
		      TransA, 
		      TransB, 
		      M, 
		      N, 
		      K, 
		      alpha, 
		      A, 
		      lda, 
		      B,
		      ldb, 
		      beta, 
		      C, 
		      N);
	}


	template<>
	void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
				    const CBLAS_TRANSPOSE TransB, 
				    const int M, 
				    const int N, 
				    const int K,
				    const double alpha, 
				    const double* A, 
				    const double* B, 
				    const double beta,
				    double* C) 
	{
	  int lda = (TransA == CblasNoTrans) ? K : M;
	  int ldb = (TransB == CblasNoTrans) ? N : K;
	  cblas_dgemm(CblasRowMajor, 
		      TransA, 
		      TransB, 
		      M, 
		      N, 
		      K, 
		      alpha, 
		      A, 
		      lda, 
		      B,
		      ldb, 
		      beta, 
		      C, 
		      N);
	}



GEMV
~~~~~~~~~~~~~~

refpath: ``src/caffe/util/math_functions.cpp``

::

	template <>
	void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, 
				   const int M,
				   const int N, 
				   const float alpha, 
				   const float* A, 
				   const float* x,
				   const float beta, 
				   float* y) 
	{
	  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
	}


	template <>
	void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, 
				    const int M,
				    const int N, 
				    const double alpha, 
				    const double* A, 
				    const double* x,
				    const double beta, 
				    double* y) 
	{
	  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
	}


Test Examples
------------------------------
