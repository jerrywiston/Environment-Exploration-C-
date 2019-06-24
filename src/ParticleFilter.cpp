#include "ParticleFilter.h"
#include "Utils.h"
#include "MotionModel.h"
#include <iostream>
#include <cassert>
#include <numeric>
#include <omp.h>

namespace gslam
{
    Particle::Particle(const Pose2D &pose, const GridMap &saved_map)
        : m_pose(pose), m_gmap(saved_map)
    {
        m_traj.push_back(pose);
    }

    Particle::~Particle()
    {
        //std::cerr<<"I am deleted!"<<std::hex<<m_traj.data();
    }

    real Particle::mapping(const BotParam &param, const SensorData &readings)
    {
        real info_gain = 0;
        auto plist = utils::EndPoints(m_pose, readings);
        for(int i=0; i<readings.sensor_size; ++i){
            if(readings.data[i] < 1)
                continue;
            else if(readings.data[i] < readings.max_dist-2)
                info_gain += m_gmap.line({m_pose[0], m_pose[1]},{plist[i][0], plist[i][1]}, true);
            else
                info_gain += m_gmap.line({m_pose[0], m_pose[1]},{plist[i][0], plist[i][1]}, false);
        }
        return info_gain;
    }

    void Particle::sampling(Control ctl, const BotParam &param, const std::array<real, 3> &sig)
    {
        MotionModel mm(0.2,0.2,0.1);
        if(ctl == Control::eForward) {
            m_pose = mm.sample(m_pose, param.velocity, 0, 0);
        } else if(ctl == Control::eBackward) {
            m_pose = mm.sample(m_pose, -param.velocity, 0, 0);
        } else if(ctl == Control::eTurnLeft) {
            m_pose = mm.sample(m_pose, 0, 0, -param.rotate_step);
        } else if(ctl == Control::eTurnRight) {
            m_pose = mm.sample(m_pose, 0, 0, param.rotate_step);
        }
        m_traj.push_back(m_pose);
    }

    void Particle::contSampling(gslam::real t, gslam::real r)
    {
        MotionModel mm(0.2,0.2,0.1);
        m_pose = mm.sample(m_pose, t, 0, r);
        m_traj.push_back(m_pose);
    }

    real Particle::nearestDistance(const Vector2 &pos, int wsize, real th) const
    {
        real min_dist = std::sqrt(wsize*wsize + wsize*wsize);
        Vector2 ans;
        real gsize = m_gmap.gridSize();
        int xx = static_cast<int>(std::round(pos[0]/gsize));
        int yy = static_cast<int>(std::round(pos[1]/gsize));
        for(int i=yy-wsize; i<yy+wsize; i++) {
            for(int j=xx-wsize; j<xx+wsize; j++) {
                if(m_gmap.getGridProb({j, i}) < th) {
                    real dist = (i-yy)*(i-yy)+(j-xx)*(j-xx);
                    if(dist < min_dist) {
                        min_dist = dist;
                        ans = {j, i};
                    }
                }
                
            }
        }
        return std::sqrt(min_dist) * gsize;
    }

    real Particle::calcLogLikelihoodField(const BotParam &param, const SensorData &readings) const
    {
        real p_hit = 0.9_r;
        real p_rand = 0.1_r;
        real sig_hit = 6.0_r;
        real q = 0.0_r;
        real inter = (param.end_angle - param.start_angle) / (param.sensor_size-1);
        for(int i=0; i<readings.data.size(); i++) {
            if(readings.data[i] > param.max_dist-1||readings.data[i]<1){
                continue;
            }
            // compute endpoints
            real theta = m_pose[2] + param.start_angle + i * inter;
            Vector2 endpoint{m_pose[0]+readings.data[i]*std::cos(utils::DegToRad(theta-90)), 
                m_pose[1]+readings.data[i]*std::sin(utils::DegToRad(theta-90))};
            
            real dist = nearestDistance(endpoint, 3, 0.2_r);
            q += std::log(p_hit * utils::GaussianPDF(0, dist, sig_hit) + p_rand/param.max_dist);            
        }
        return q;
    }

    ParticleFilter::ParticleFilter(const Pose2D &pose, const BotParam &param, const GridMap &saved_map, const int size)
        : m_param(param), m_size(size)
    {
        m_particles.reserve(m_size);
        m_weights.reserve(m_size);
        m_infoGain.reserve(m_size);
        for(int i=0; i<m_size; ++i){
            m_particles.emplace_back(pose, saved_map);
            m_weights.push_back(1.0 / (float)m_size);
            m_infoGain.push_back(0);
        }
    }

    void Particle::mappingList(const BotParam &param)
    {
        for(int j=0; j<m_obsList.size(); ++j){
            auto plist = utils::EndPoints(m_posList[j], m_obsList[j]);
            for(int i=0; i<m_obsList[j].sensor_size; ++i){
                if(m_obsList[j].data[i] > m_obsList[j].max_dist-1 || m_obsList[j].data[i] < 1)
                    continue;
                m_gmap.line({m_posList[j][0], m_posList[j][1]},{plist[i][0], plist[i][1]}, true);
            }
        }
    }

    real ParticleFilter::feed(Control ctl, const SensorData &readings)
    {
        std::vector<real> field(m_size);
        real n_tmp = 0;
        m_mapCount++;
        #pragma omp parallel for num_threads(8)
        for(int i=0; i<m_size; i++) {
            // Update particle location
            m_particles[i].sampling(ctl, m_param);
            field[i] = m_particles[i].calcLogLikelihoodField(m_param, readings);
            m_infoGain[i] = m_particles[i].mapping(m_param, readings);
            
            //m_particles[i].addObs(readings);
            //if(m_mapCount > 2){
            //    m_particles[i].mappingList(m_param);
            //    m_particles[i].clearObs();
            //    m_mapCount = 0;
            //}
        }
        // normalize of field array is not needed here
        real normalize_max = -9999;
        for(int i=0; i<m_size; ++i){
            if(field[i] > normalize_max)
                normalize_max = field[i];
        }

        for(int i=0; i<m_size; ++i)
            m_weights[i] = std::exp(field[i] - normalize_max);

        real tmp = 0;
        for(int i=0; i<m_size; ++i)
            n_tmp += m_weights[i];
        
        if(n_tmp != 0){
            for(int i=0; i<m_size; ++i)
                m_weights[i] = m_weights[i] / n_tmp;
        }

        // Calculate Neff
        real Neff = 0;
        for(int i=0; i<m_size; ++i){
            Neff += m_weights[i] * m_weights[i];
        }
        Neff = 1.0 / Neff;
        return Neff / m_size;
    }

    real ParticleFilter::contFeed(gslam::real t, gslam::real r, const SensorData &readings)
    {
        std::vector<real> field(m_size);
        real n_tmp = 0;
        m_mapCount++;
        #pragma omp parallel for num_threads(8)
        for(int i=0; i<m_size; i++) {
            // Update particle location
            m_particles[i].contSampling(t, r);
            field[i] = m_particles[i].calcLogLikelihoodField(m_param, readings);
            m_infoGain[i] = m_particles[i].mapping(m_param, readings);
            
            //m_particles[i].addObs(readings);
            //if(m_mapCount > 2){
            //    m_particles[i].mappingList(m_param);
            //    m_particles[i].clearObs();
            //    m_mapCount = 0;
            //}
        }
        // normalize of field array is not needed here
        real normalize_max = -9999;
        for(int i=0; i<m_size; ++i){
            if(field[i] > normalize_max)
                normalize_max = field[i];
        }

        for(int i=0; i<m_size; ++i)
            m_weights[i] = std::exp(field[i] - normalize_max);

        real tmp = 0;
        for(int i=0; i<m_size; ++i)
            n_tmp += m_weights[i];
        
        if(n_tmp != 0){
            for(int i=0; i<m_size; ++i)
                m_weights[i] = m_weights[i] / n_tmp;
        }

        // Calculate Neff
        real Neff = 0;
        for(int i=0; i<m_size; ++i){
            Neff += m_weights[i] * m_weights[i];
        }
        Neff = 1.0 / Neff;
        return Neff / m_size;
    }

    void ParticleFilter::resampling()
    {
        std::discrete_distribution<int> distribution{m_weights.cbegin(), m_weights.cend()};
        std::vector<Particle> new_particles;
        new_particles.reserve(m_particles.size());
        for(int i=0; i<m_size; i++) {
            int id = distribution(m_generator);
            new_particles.push_back(m_particles[id]);
        }
        m_particles = new_particles;
    }
}