#include "ParticleFilter.h"
#include "Utils.h"
#include "MotionModel.h"
#include <iostream>
#include <cassert>
#include <numeric>
#include <omp.h>
#include <math.h>

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

    void Particle::Sampling(gslam::real t, gslam::real r, gslam::real noise_nor, gslam::real noise_tan, gslam::real noise_ang)
    {
        MotionModel mm(noise_nor, noise_tan, noise_ang);
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

    real ParticleFilter::feed(gslam::real t, gslam::real r, const SensorData &readings)
    {
        std::vector<real> field(m_size);
        real n_tmp = 0;
        m_mapCount++;
        #pragma omp parallel for num_threads(8)
        for(int i=0; i<m_size; i++) {
            // Update particle location
            m_particles[i].Sampling(t, r, m_param.noise_nor, m_param.noise_tan, m_param.noise_ang);
            field[i] = m_particles[i].calcLogLikelihoodField(m_param, readings);
            m_infoGain[i] = m_particles[i].mapping(m_param, readings);
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

    real ParticleFilter::getMapInfoGain(){
        real ig = 0;
        for(int i=0; i<m_size; ++i){
            ig += m_infoGain[i] * m_weights[i];
        }
        return ig;
    }
    
    real ParticleFilter::getTrajEntropy(){
        real traj_ent = 0;
        int tsize = (int)m_particles[0].getTraj().size();
        for(int j=1; j<tsize; ++j){
            gslam::Pose2D mean(0,0,0); 
            gslam::Pose2D var(0,0,0);
           
            for(int i=0; i<m_size; ++i)
                mean += m_weights[i] * m_particles[i].getTraj()[j];
             
            for(int i=0; i<m_size; ++i){
                Pose2D temp = m_particles[i].getTraj()[j] - mean;
                var[0] += m_weights[i] * temp[0] * temp[0];
                var[1] += m_weights[i] * temp[1] * temp[1];
                var[2] += m_weights[i] * temp[2] * temp[2];
            }          
            
            // pi=3.1415926, e=2.71828, h(x) = 0.5 * log(2*pi*e*var)
            traj_ent +=  0.5 * log(2 * 3.142 * 2.718 * (var[0] + var[1] + var[2]) + 1e-3); 
        }

        traj_ent /= (tsize-1);
        return traj_ent;
    }
    
}