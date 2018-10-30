#ifndef _GRIDSLAM_CONFIG_H_
#define _GRIDSLAM_CONFIG_H_

#include <Eigen/Core>

namespace gslam {
    using Vector2i = Eigen::Vector2i;
    using Vector2f = Eigen::Vector2f;
    using Vector2d = Eigen::Vector2d;

    using Vector3i = Eigen::Vector3i;
    using Vector3f = Eigen::Vector3f;
    using Vector3d = Eigen::Vector3d;
    
#ifdef USE_DOUBLE
    using real = double;
    using Vector2 = Vector2d;
    #define CV_REAL_C1 CV_64FC1
#else
    using real = float;
    using Vector2 = Vector2f;
    using Vector3 = Vector3f;
    #define CV_REAL_C1 CV_32FC1
#endif

}

#define MEGAGRID_SIZE 33

static inline gslam::real operator"" _r(long double val)  
{  
    return static_cast<gslam::real>(val);
}  

#endif