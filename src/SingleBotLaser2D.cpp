#include "SingleBotLaser2D.h"
#include "Utils.h"
#include "MotionModel.h"
#include <Eigen/Eigen>
#include <cmath>
#include <iostream>

namespace gslam
{
    SingleBotLaser2DGrid::SingleBotLaser2DGrid(const Pose2D &bot_pose, const BotParam &param, const Storage2D<uint8_t> &map)
        : m_pose(bot_pose), m_param(param)
    {
        m_imageMap = MatrixXf::Zero(map.rows(), map.cols());
        
        auto data = map.data();
        // TODO: Use Eigen function instead...
        for(int i=0; i<map.rows(); ++i) {
            for(int j=0; j<map.cols(); ++j){
                m_imageMap(i,j) = *(data++) / 255.0f;
            }
        }

        m_traj.push_back(bot_pose);
    }

    real SingleBotLaser2DGrid::rayCast(const Pose2D &pose) const
    {
        Vector2i origin={std::round(pose[0]), std::round(pose[1])};
        Vector2i end={std::round(pose[0]+m_param.max_dist*cos(utils::Deg2Rad(pose[2]))),
                      std::round(pose[1]+m_param.max_dist*sin(utils::Deg2Rad(pose[2])))};
        
        std::vector<Vector2i> plist;
        utils::Bresenham(plist, origin, end);
        real dist = m_param.max_dist;
        for(int i=0; i<plist.size(); ++i){
            if(plist[i][1]>=m_imageMap.rows() || plist[i][0]>=m_imageMap.cols() || plist[i][1]<0 || plist[i][0]<0)
                continue;
            if(m_imageMap(plist[i][1], plist[i][0])<0.8){
                real tmp = std::pow(float(plist[i][0]) - pose[0], 2) + std::pow(float(plist[i][1]) - pose[1], 2);
                tmp = std::sqrt(tmp);
                if(tmp < dist)
                    dist = tmp;
            }
        }
        return dist;
    }

    SensorData SingleBotLaser2DGrid::scan() const{
        real inter = (m_param.end_angle - m_param.start_angle) / (m_param.sensor_size-1);
        SensorData sdata;
        sdata.sensor_size = m_param.sensor_size;
        sdata.start_angle = m_param.start_angle;
        sdata.end_angle = m_param.end_angle;
        sdata.max_dist = m_param.max_dist;
        sdata.data.resize(m_param.sensor_size);

        for(int i=0; i<m_param.sensor_size; ++i){
            real theta = m_pose[2] + m_param.start_angle + i*inter;
            sdata.data[i] = rayCast({m_pose[0], m_pose[1], theta-90});
        }
        return sdata;
    }

    bool SingleBotLaser2DGrid::botAction(real t, real r){
        MotionModel mm(m_param.noise_nor, m_param.noise_tan, m_param.noise_ang);
        Pose2D org_pose = m_pose;
        m_pose = mm.sample(m_pose, t, 0, r);
        std::vector<Vector2i> rec;
        utils::Bresenham(rec, {org_pose[0], org_pose[1]}, {m_pose[0], m_pose[1]});
        for(auto &r: rec) {
            
            if(r[1] < 0 || r[1] > m_imageMap.rows() || r[0] < 0 || r[0] > m_imageMap.cols()) {
                // Out of range, restore pose!
                m_pose = org_pose;
                return false;
            }
            if(m_imageMap(r[1], r[0]) < 0.5f) {
                // blocked by something, restore pose!
                m_pose = org_pose;
                return false;
            }
        }

        m_traj.push_back(m_pose);
        return true;
    }
}