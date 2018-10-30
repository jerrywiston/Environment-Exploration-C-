#ifndef _GRIDSLAM_SINGLEBOTLASER2D_H_
#define _GRIDSLAM_SINGLEBOTLASER2D_H_

#include "Config.h"
#include "Type.h"
#include <vector>
#include <string>
#include <Eigen/Eigen>

namespace gslam
{
    enum class Control {
        eNone=0,
        eForward=1,
        eBackward=2,
        eTurnLeft=3,
        eTurnRight=4,

        eDown=5,
        eLeft=6,
        eRight=7,
        eUp=8
    };

    class SingleBotLaser2DGrid {
    public:
        SingleBotLaser2DGrid(const Vector2 &bot_pos, const real theta, const BotParam &param,
            const std::string &fname);

        real rayCast(const Vector2 &pos, const real theta) const;

        SensorData scan() const;
        void botAction(Control action);

        // Get Function
        Eigen::MatrixXf getMap(){
            return m_imageMap;
        }

        Vector2 getPos(){
            return m_pos;
        }

        real getTheta(){
            return m_theta;
        }

        BotParam getParam(){
            return m_param;
        }
    private:
        Eigen::MatrixXf m_imageMap;
        Vector2 m_pos;
        real m_theta;
        BotParam m_param;
    };
}

#endif