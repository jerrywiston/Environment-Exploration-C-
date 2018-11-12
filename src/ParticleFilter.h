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
        real calcLogLikelihoodField(const BotParam &param, const SensorData &readings) const;
        
        Vector3 getPose(){
            return m_pose;
        }

        GridMap getMap(){
            return m_gmap;
        }

        std::vector<Vector3> getTraj(){
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
        Vector3 m_pose;
        GridMap m_gmap;
        std::vector<Vector3> m_traj;  
        real nearestDistance(const Vector2 &pos, int wsize, real th) const;

        std::vector<SensorData> m_obsList;
        std::vector<Vector3> m_posList;
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

        std::vector<Vector3> getTraj(int id){
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