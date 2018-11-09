#ifndef _GRIDSLAM_PARTICLEFILTER_H_
#define _GRIDSLAM_PARTICLEFILTER_H_

#include "Config.h"
#include "GridMap.h"
#include "SingleBotLaser2D.h"
#include "MotionModel.h"
#include <random>

namespace gslam
{

    class Particle {
    public:
        Particle(const Vector3 &pose, const GridMap &saved_map);
        void mapping(const BotParam &param, const SensorData &reading);
        void sampling(Control ctl, const BotParam &param, const std::array<real, 3> &sig={0.4_r, 0.4_r, 0.4_r});
        real calcLikelihood(const BotParam &param, const SensorData &readings) const;
        
        Vector3 getPose(){
            return m_pose;
        }

        GridMap getMap(){
            return m_gmap;
        }
    private:
        Vector3 m_pose;
        GridMap m_gmap;
        real nearestDistance(const Vector2 &pos, int wsize, real th) const;
    };

    class ParticleFilter {
    public:
        ParticleFilter(const Vector3 &pose, const BotParam &param, const GridMap &saved_map, const int size);
        real feed(Control ctl, const SensorData &readings);
        void resampling();
        
        int getSize(){
            return m_size;
        }
        
        Vector3 getPose(int id){
            return m_particles[id].getPose();
        }

        Particle getParticle(int id){
            return m_particles[id];
        }
    
    private:
        int m_size;
        std::vector<real> m_weights;
        BotParam m_param;
        std::vector<Particle> m_particles;
        std::default_random_engine m_generator;
  
    };
}

#endif