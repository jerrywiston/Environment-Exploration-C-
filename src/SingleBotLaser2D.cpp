#include "SingleBotLaser2D.h"
#include "Utils.h"
#include <Eigen/Eigen>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>

namespace gslam
{
    SingleBotLaser2DGrid::SingleBotLaser2DGrid(const Vector2 &bot_pos, const real theta, const BotParam &param, const std::string &fname)
        : m_pos(bot_pos), m_theta(theta), m_param(param)
    {
        cv::Mat image;
        image = cv::imread(fname, cv::IMREAD_GRAYSCALE);
        m_imageMap = Eigen::MatrixXf::Zero(image.size().height, image.size().width);

        for(int i=0; i<image.size().height; ++i)
            for(int j=0; j<image.size().width; ++j){
                m_imageMap(i,j) = (real)image.at<uchar>(i,j) / 255.0_r;
            }
    }

    real SingleBotLaser2DGrid::rayCast(const Vector2 &pos, const real theta) const
    {
        Vector2i origin={std::round(pos[0]), std::round(pos[1])};
        Vector2i end={std::round(pos[0]+m_param.max_dist*cos(utils::Deg2Rad(theta))),
                      std::round(pos[1]+m_param.max_dist*sin(utils::Deg2Rad(theta)))};
        
        std::vector<Vector2i> plist;
        utils::Bresenham(plist, origin, end);
        real dist = m_param.max_dist;
        for(int i=0; i<plist.size(); ++i){
            if(plist[i][1]>=m_imageMap.rows() || plist[i][0]>=m_imageMap.cols() || plist[i][1]<0 || plist[i][0]<0)
                continue;
            if(m_imageMap(plist[i][1], plist[i][0])<0.8){
                real tmp = std::pow(float(plist[i][0]) - pos[0], 2) + std::pow(float(plist[i][1]) - pos[1], 2);
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
            real theta = m_theta + m_param.start_angle + i*inter;
            sdata.data[i] = rayCast(m_pos,theta);
        }
        return sdata;
    }

    void SingleBotLaser2DGrid::botAction(Control action){
        real vx = sin(utils::Deg2Rad(m_theta));
        real vy = cos(utils::Deg2Rad(m_theta));
        switch(action){
            case Control::eForward:
                m_pos[0] -= m_param.velocity*vx;
                m_pos[1] += m_param.velocity*vy;
                break;

            case Control::eBackward:
                m_pos[0] += m_param.velocity*vx;
                m_pos[1] -= m_param.velocity*vy;
                break;
            
            case Control::eTurnLeft:
                m_theta -= m_param.rotate_step;
                if(m_theta >= 360)
                    m_theta -= 360;
                else if(m_theta <0)
                    m_theta += 360;
                break;

            case Control::eTurnRight:
                m_theta += m_param.rotate_step;
                if(m_theta >= 360)
                    m_theta -= 360;
                else if(m_theta <0)
                    m_theta += 360;
                break;
        }
    }
}