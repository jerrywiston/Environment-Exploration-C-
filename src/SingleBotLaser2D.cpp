#include "SingleBotLaser2D.h"
#include "Utils.h"
#include "MotionModel.h"
#include <Eigen/Eigen>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>

namespace gslam
{
    SingleBotLaser2DGrid::SingleBotLaser2DGrid(const Vector3 &bot_pose, const BotParam &param, const std::string &fname)
        : m_pose(bot_pose), m_param(param)
    {
        cv::Mat image;
        image = cv::imread(fname, cv::IMREAD_GRAYSCALE);
        m_imageMap = Eigen::MatrixXf::Zero(image.size().height, image.size().width);

        for(int i=0; i<image.size().height; ++i)
            for(int j=0; j<image.size().width; ++j){
                m_imageMap(i,j) = (real)image.at<uchar>(i,j) / 255.0_r;
            }
        
        m_traj.push_back(bot_pose);
    }

    real SingleBotLaser2DGrid::rayCast(const Vector3 &pose) const
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

    void SingleBotLaser2DGrid::botAction(Control action){
        MotionModel mm(0.5,0.5,0.3);
        switch(action){
            case Control::eForward:
                m_pose = mm.sample(m_pose, m_param.velocity, 0, 0);
                break;

            case Control::eBackward:
                m_pose = mm.sample(m_pose, -m_param.velocity, 0, 0);
                break;
            
            case Control::eTurnLeft:
                m_pose = mm.sample(m_pose, 0, 0, -m_param.rotate_step);
                break;

            case Control::eTurnRight:
                m_pose = mm.sample(m_pose, 0, 0, m_param.rotate_step);
                break;
        }
        m_traj.push_back(m_pose);
    }
}