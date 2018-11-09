#ifndef _GRIDSLAM_UTILS_H_
#define _GRIDSLAM_UTILS_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include "Config.h"
#include "Type.h"
#include <string>

namespace gslam
{
    namespace utils
    {
        void VisualizeGrid(const cv::Mat &mat, const std::string name);
        void Bresenham(std::vector<Vector2i> &rec, const Vector2i &xy1, const Vector2i &xy2);
        real Gaussian(real mu, real sigma);
        real GaussianPDF(real x, real mu, real sigma);
        static inline real DegToRad(real deg)
        { return deg*3.1415926_r/180.0_r; }
        std::vector<Vector2> EndPoints(Vector3 pose, SensorData sdata);
        real Deg2Rad(real deg);
    }
}

#endif