#include "MotionModel.h"
#include "Utils.h"

namespace gslam
{
    MotionModel::MotionModel(const real nor_var, const real tan_var, const real ang_var)
        : normal_var(nor_var), tangent_var(tan_var), angular_var(ang_var)
    {

    }

    Vector3 MotionModel::sample(const Vector3 &pose, const real n, const real t, const real theta)
    {
        real n_sample = utils::Gaussian(n, normal_var);
        real t_sample = utils::Gaussian(t, tangent_var);
        real th_sample = utils::Gaussian(theta, angular_var);

        real ang_tmp = pose[2] + th_sample;
        Vector3 pose_samp = {
            pose[0] + n_sample*cos(utils::Deg2Rad(ang_tmp)) + t_sample*sin(utils::Deg2Rad(ang_tmp)),
            pose[1] + n_sample*sin(utils::Deg2Rad(ang_tmp)) + t_sample*cos(utils::Deg2Rad(ang_tmp)),
            ang_tmp
        };
        return pose_samp;
    }
}