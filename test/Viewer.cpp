#include "Viewer.h"
#include "Utils.h"
#include <Eigen/Eigen>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

namespace gslam
{
    Viewer::Viewer(){

    }

    cv::Mat Viewer::Map2Image(Eigen::MatrixXf &m){
        cv::Mat img(m.rows(), m.cols(), CV_8UC3, cv::Scalar(0,0,0));
        for(int i=0; i<m.rows(); ++i)
            for(int j=0; j<m.cols(); ++j){
                int value = int(m(i,j)*255);
                img.at<cv::Vec3i>(i,j,0) = value;
                img.at<cv::Vec3i>(i,j,1) = value;
                img.at<cv::Vec3i>(i,j,2) = value;
            }
        return img;
    }

}