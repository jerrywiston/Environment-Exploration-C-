#include "ParticleFilter.h"
#include "Utils.h"
#include "MotionModel.h"
#include <cassert>
#include <numeric>
#include <omp.h>

namespace gslam
{
    Particle::Particle(const Vector3 &pose, const GridMap &saved_map)
        : m_pose(pose), m_gmap(saved_map)
    {

    }

    void Particle::mapping(const BotParam &param, const SensorData &readings)
    {
        auto plist = utils::EndPoints(m_pose, readings);
        for(int i=0; i<readings.sensor_size; ++i){
            if(readings.data[i] > readings.max_dist-1 || readings.data[i] < 1)
                continue;
            m_gmap.line({m_pose[0], m_pose[1]},{plist[i][0], plist[i][1]});
        }
    }

    void Particle::sampling(Control ctl, const BotParam &param, const std::array<real, 3> &sig)
    {
        MotionModel mm(0.5,0.5,0.5);
        if(ctl == Control::eForward) {
            m_pose = mm.sample(m_pose, param.velocity, 0, 0);
        } else if(ctl == Control::eBackward) {
            m_pose = mm.sample(m_pose, -param.velocity, 0, 0);
        } else if(ctl == Control::eTurnLeft) {
            m_pose = mm.sample(m_pose, 0, 0, -param.rotate_step);
        } else if(ctl == Control::eTurnRight) {
            m_pose = mm.sample(m_pose, 0, 0, param.rotate_step);
        }
    }

    real Particle::nearestDistance(const Vector2 &pos, int wsize, real th) const
    {
        real min_dist = 9999;
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
        return std::sqrt(min_dist*gsize);
    }

    real Particle::calcLikelihood(const BotParam &param, const SensorData &readings) const
    {
        real p_hit = 0.9_r;
        real p_rand = 0.1_r;
        real sig_hit = 10.0_r;
        real q = 1.0_r;
        real inter = (param.end_angle - param.start_angle) / (param.sensor_size-1);
        for(int i=0; i<readings.data.size(); i++) {
            if(readings.data[i] > param.max_dist-1||readings.data[i]<1)
                continue;
            // compute endpoints
            real theta = m_pose[2] + param.start_angle + i * inter;
            Vector2 endpoint{m_pose[0]+readings.data[i]*std::cos(utils::DegToRad(theta-90)), 
                m_pose[1]+readings.data[i]*std::sin(utils::DegToRad(theta-90))};
            
            real dist = nearestDistance(endpoint, 4, 0.2_r);
            q = 10 * q * (p_hit * utils::GaussianPDF(0, dist, sig_hit) + p_rand/param.max_dist);
        }
        return q;
    }

    ParticleFilter::ParticleFilter(const Vector3 &pose, const BotParam &param, const GridMap &saved_map, const int size)
        : m_param(param), m_size(size)
    {
        Particle p(pose, saved_map);
        for(int i=0; i<m_size; ++i){
            m_particles.push_back(p);
            m_weights.push_back(1.0 / (float)m_size);
        }
    }

    real ParticleFilter::feed(Control ctl, const SensorData &readings)
    {
        std::vector<real> field;
        real n_tmp = 0;
        #pragma omp parallel for
        for(int i=0; i<m_size; i++) {
            // Update particle location
            m_particles[i].sampling(ctl, m_param);
            field.push_back(m_particles[i].calcLikelihood(m_param, readings));
            m_particles[i].mapping(m_param, readings);
        }
        // normalize of field array is not needed here
        for(int i=0; i<m_size; ++i)
            n_tmp += field[i];
        
        if(n_tmp != 0){
            for(int i=0; i<m_size; ++i)
                m_weights[i] = field[i] / n_tmp;
        }

        // Calculate Neff
        real Neff = 0;
        for(int i=0; i<m_size; ++i){
            Neff += m_weights[i] * m_weights[i];
        }
        Neff = 1.0 / Neff;
        //std::cout << ">>>>>" << std::endl;
        //for(int i=0; i<m_size; ++i)
        //    std::cout << m_weights[i] << " ";
        //std::cout << "<<<<<" << std::endl;
        return Neff / m_size;
    }

    void ParticleFilter::resampling()
    {
        //for(int i=0; i<m_size; ++i)
        //    std::cout << m_weights[i] << " ";
        //std::cout << std::endl;
        std::discrete_distribution<int> distribution{m_weights.cbegin(), m_weights.cend()};
        std::vector<Particle> new_particles;
        new_particles.reserve(m_particles.size());
        for(int i=0; i<m_size; i++) {
            int id = distribution(m_generator);
            //std::cout << id << " ";
            new_particles.push_back(m_particles[id]);
        }
        //std::cout << std::endl;
        m_particles = new_particles;
    }
}