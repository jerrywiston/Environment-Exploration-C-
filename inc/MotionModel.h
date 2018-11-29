#ifndef _GRIDSLAM_MOTIONMODEL_H_
#define _GRIDSLAM_MOTIONMODEL_H_

#include "Utils.h"
#include <Eigen/Eigen>

namespace gslam
{
    class MotionModel{
    public:
        MotionModel(const real nor_var, const real tan_var, const real ang_var);
        Vector3 sample(const Vector3 &pose, const real n, const real t, const real theta);

    private:
        real normal_var;
        real tangent_var;
        real angular_var;
    };
}

#endif