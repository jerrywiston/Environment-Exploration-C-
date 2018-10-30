#ifndef _GRIDSLAM_PARTICLEFILTER_H_
#define _GRIDSLAM_PARTICLEFILTER_H_

#include "Config.h"
#include "GridMap.h"
#include "SingleBotLaser2D.h"
#include <random>

namespace gslam
{

    class Particle {
    public:
        Particle(const Vector2 &pos, const GridMap &saved_map);
        void mapping(const BotParam &param, const SensorData &reading);
        void sampling(Control ctl, const BotParam &param, const std::array<real, 3> &sig={0.4_r, 0.4_r, 0.4_r});
        real calcLikelihood(const BotParam &param, const SensorData &readings) const;
    private:
        Vector2 m_pos;
        real m_theta;
        GridMap m_gmap;
        real nearestDistance(const Vector2 &pos, int wsize, real th) const;
    };

    class ParticleFilter {
    public:
        ParticleFilter(const Vector2 &pos, const BotParam &param, const GridMap &saved_map);
        void feed(Control ctl, const SensorData &readings);
    private:
        void resampling(const SensorData &, const std::vector<real> &weights);
        BotParam m_param;
        std::vector<Particle> m_particles;
        std::default_random_engine m_generator;
  
    };
}

#endif