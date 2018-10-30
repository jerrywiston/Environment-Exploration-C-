#ifndef _GRIDSLAM_VIEWER_H_
#define _GRIDSLAM_VIEWER_H_

#include "Config.h"
#include "Type.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace gslam{
    class Viewer{
    public:
        Viewer();
        cv::Mat Map2Image(Eigen::MatrixXf &m);

    private:
    };
}

#endif