#include "ParticleFilter.h"
#include "Utils.h"
#include <cassert>
#include <numeric>

namespace gslam
{
    Particle::Particle(const Vector2 &pos, const GridMap &saved_map)
        : m_pos(pos), m_gmap(saved_map)
    {

    }

    void Particle::mapping(const BotParam &param, const SensorData &readings)
    {
        real inter = (param.end_angle - param.start_angle) / (param.sensor_size-1);
        for(int i=0; i<param.sensor_size; i++) {
            if(readings.data[i] > param.max_dist-1||readings.data[i]<1)
                continue;
            real theta = m_theta + param.start_angle + i * inter;
            m_gmap.line(
                m_pos,
                {m_pos[0]+readings.data[i]*std::cos(utils::DegToRad(theta)), 
                    m_pos[1]+readings.data[i]*std::sin(utils::DegToRad(theta))}
            );
        }
    }

    void Particle::sampling(Control ctl, const BotParam &param, const std::array<real, 3> &sig)
    {
        Vector2 vec{std::sin(utils::DegToRad(m_theta)), std::cos(utils::DegToRad(m_theta))};
        real vel = param.velocity;
        real ang = param.rotate_step;
        if(ctl == Control::eForward) {
            m_pos += Vector2(-vel*vec[0], vel*vec[1]);
        } else if(ctl == Control::eBackward) {
            m_pos += Vector2(vel*vec[0], -vel*vec[1]);
        } else if(ctl == Control::eTurnLeft) {
            m_theta += ang;
            m_theta = std::fmod(m_theta, 360.0_r);
        } else if(ctl == Control::eTurnRight) {
            m_theta -= ang;
            m_theta = std::fmod(m_theta + 360.0_r, 360.0_r);
        }

        m_pos += Vector2(utils::Gaussian(0.0, sig[0]), utils::Gaussian(0.0, sig[1]));
        m_theta += utils::Gaussian(0.0, sig[2]);
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
        real sig_hit = 3.0_r;
        real q = 1.0_r;
        real inter = (param.end_angle - param.start_angle) / (param.sensor_size-1);
        for(int i=0; i<readings.data.size(); i++) {
            if(readings.data[i] > param.max_dist-1||readings.data[i]<1)
                continue;
            // compute endpoints
            real theta = m_theta + param.start_angle + i * inter;
            Vector2 endpoint{m_pos[0]+readings.data[i]*std::cos(utils::DegToRad(theta)), 
                m_pos[1]+readings.data[i]*std::sin(utils::DegToRad(theta))};
            
            real dist = nearestDistance(endpoint, 4, 0.2_r);
            q = q * (p_hit * utils::GaussianPDF(0, dist, sig_hit));
        }
        return q;
    }

    void ParticleFilter::feed(Control ctl, const SensorData &readings)
    {
        std::vector<real> field(0.0_r, m_particles.size());
        for(int i=0; i<m_particles.size(); i++) {
            // Update particle location
            m_particles[i].sampling(ctl, m_param);
            field[i] = m_particles[i].calcLikelihood(m_param, readings);
        }
        // normalize of field array is not needed here
        resampling(readings, field);
    }

    void ParticleFilter::resampling(const SensorData &readings, const std::vector<real> &weights)
    {
        assert(weights.size() == m_particles.size());

        std::discrete_distribution<int> distribution{weights.cbegin(), weights.cend()};
        std::vector<int> map_rec(0, m_particles.size());
        std::vector<Particle> new_particles;
        new_particles.reserve(m_particles.size());
        for(int i=0; i<m_particles.size(); i++) {
            int id = distribution(m_generator);
            if(!map_rec[id]) {
                // only map high weight particle
                m_particles[id].mapping(m_param, readings);
                map_rec[id] = 1;
            }
            new_particles.push_back(m_particles[id]);
        }
        m_particles = new_particles;
    }
}