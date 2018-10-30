#ifndef _GRIDSLAM_TYPE_H_
#define _GRIDSLAM_TYPE_H_

#include "Config.h"
#include <vector>

namespace gslam {
    struct BotParam {
        int sensor_size;
        // in degree
        real start_angle;
        real end_angle;
        real max_dist;

        // odom
        real velocity;
        // degree
        real rotate_step;
    };

    struct SensorData {
        int sensor_size;
        real start_angle;
        real end_angle;
        real max_dist;
        std::vector<real> data;
    };
}

#endif