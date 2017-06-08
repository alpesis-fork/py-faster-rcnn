/*

   Usage:

      $ ./build/examples/fasterrcnn/fasterrcnn.bin \
        models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt \
        data/fasterrcnn/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel \
        data/fasterrcnn/demo/000456.jpg

*/


#include <string>
#include <iostream>

#include "caffe/caffe.hpp"

using namespace caffe;


class Detector
{
    public:
        Detector (const std::string& model_file,
                  const std::string& trained_file);
        ~Detector ();

    private:
        shared_ptr<Net<float> > net_;

};


Detector::Detector (const std::string& model_file,
                    const std::string& trained_file)
{
    std::cout << "Detector: " << std::endl;

    #ifdef CPU_ONLY
        Caffe::set_mode (Caffe::CPU);
    #else
        Caffe::set_mode (Caffe::GPU);
    #endif

    // load the network
    net_.reset (new Net<float>(model_file, TEST));
}


Detector::~Detector ()
{

}


int main (int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " xxx.prototxt xxx.caffemodel"
                  << " image.jpg" << std::endl;
        return 1;
    }


    std::string model_file = argv[1];
    std::string trained_file = argv[2];
    std::string file = argv[3];

    std::cout << "Faster RCNN: " << std::endl;
    Detector detector (model_file, trained_file);    

    return 0;
}
