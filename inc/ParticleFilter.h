#ifndef _GRIDSLAM_PARTICLEFILTER_H_
#define _GRIDSLAM_PARTICLEFILTER_H_

#include "Config.h"
#include "GridMap.h"
#include "SingleBotLaser2D.h"
#include "MotionModel.h"
#include <random>
#include <array>

namespace gslam
{

    class Particle {
    public:
        Particle(const Pose2D &pose, const GridMap &saved_map);
        ~Particle();
        void mapping(const BotParam &param, const SensorData &reading);
        void sampling(Control ctl, const BotParam &param, const std::array<real, 3> &sig={0.4_r, 0.4_r, 0.4_r});
        real calcLogLikelihoodField(const BotParam &param, const SensorData &readings) const;
        
        Pose2D getPose(){
            return m_pose;
        }

        GridMap &getMap(){
            return m_gmap;
        }

        std::vector<Pose2D> &getTraj(){
            return m_traj;
        }

        void mappingList(const BotParam &param);

        void addObs(const SensorData &reading){
            m_obsList.push_back(reading);
            m_posList.push_back(m_pose);
        }

        void clearObs(){
            m_obsList.clear();
            m_posList.clear();
        }
    private:
        Pose2D m_pose;
        GridMap m_gmap;
        std::vector<Pose2D> m_traj;  
        real nearestDistance(const Vector2 &pos, int wsize, real th) const;

        std::vector<SensorData> m_obsList;
        std::vector<Pose2D> m_posList;
    };

    class ParticleFilter {
    public:
        ParticleFilter(const Pose2D &pose, const BotParam &param, const GridMap &saved_map, const int size);
        real feed(Control ctl, const SensorData &readings);
        void resampling();
        
        int getSize(){
            return m_size;
        }
        
        Vector3 getPose(int id){
            return m_particles[id].getPose();
        }

        // We don't use reference since the particle may be removed.
        Particle &getParticle(int id){
            return m_particles[id];
        }

        // this function will be removed
        std::vector<Pose2D> getTraj(int id){
            return m_particles[id].getTraj();
        }

        int bestSampleId(){
            real tmp = -1000;
            int id = 0;
            for(int i=0; i<m_weights.size(); ++i){
                if(m_weights[i]>tmp){
                    tmp = m_weights[i];
                    id = i;
                }
            }
            return id;
        }
    
    private:
        int m_size;
        std::vector<real> m_weights;
        BotParam m_param;
        std::vector<Particle> m_particles;
        std::default_random_engine m_generator;
        int m_mapCount = 0;
    };
}

#endif