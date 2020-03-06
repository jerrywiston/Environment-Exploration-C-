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
    struct IdRecord{
        int id;
        int timestemp;
        IdRecord(int i, int t): id(i), timestemp(t){};
    };

    class Particle {
    public:
        Particle(const Pose2D &pose, const GridMap &saved_map);
        ~Particle();
        real mapping(const BotParam &param, const SensorData &reading);
        void Sampling(gslam::real t, gslam::real r, gslam::real noise_nor, gslam::real noise_tan, gslam::real noise_ang);
        real calcLogLikelihoodField(const BotParam &param, const SensorData &readings) const;
        
        Pose2D getPose(){
            return m_pose;
        }

        GridMap &getMap(){
            return m_gmap;
        }

        std::vector<IdRecord> &getIdRecord(){
            return m_idRecord;
        }

        std::vector<Pose2D> &getTraj(){
            return m_traj;
        }

        void markId(int id, int timestemp){
            IdRecord temp(id, timestemp);
            m_idRecord.push_back(temp);
        }

    private:
        Pose2D m_pose;
        GridMap m_gmap;
        std::vector<Pose2D> m_traj;  
        real nearestDistance(const Vector2 &pos, int wsize, real th) const;

        std::vector<SensorData> m_obsList;
        std::vector<Pose2D> m_posList;
        std::vector<IdRecord> m_idRecord;
    };

    class ParticleFilter {
    public:
        ParticleFilter(const Pose2D &pose, const BotParam &param, const GridMap &saved_map, const int size);
        real feed(gslam::real t, gslam::real r, const SensorData &readings);
        void resampling();
        
        real getMapInfoGain();
        real getTrajEntropy();
        
        int getSize(){
            return m_size;
        }
        
        Vector3 getPose(int id){
            return m_particles[id].getPose();
        }

        real getWeight(int id){
            return m_weights[id];
        }

        real getInfoGain(int id){
            return m_infoGain[id];
        }

        // We don't use reference since the particle may be removed.
        Particle &getParticle(int id){
            return m_particles[id];
        }

        // this function will be removed
        std::vector<Pose2D> getTraj(int id){
            return m_particles[id].getTraj();
        }

        std::vector<IdRecord> getIdRecord(int id){
            return m_particles[id].getIdRecord();
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

        void markParticles(int timestemp){
            for(int i=0; i<m_particles.size(); ++i)
                m_particles[i].markId(i,timestemp);
        }

        
    
    private:
        int m_size;
        std::vector<real> m_weights;
        std::vector<real> m_infoGain;
        BotParam m_param;
        std::vector<Particle> m_particles;
        std::default_random_engine m_generator;
        int m_mapCount = 0;
    };
}

#endif