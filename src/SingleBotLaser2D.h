#ifndef _GRIDSLAM_SINGLEBOTLASER2D_H_
#define _GRIDSLAM_SINGLEBOTLASER2D_H_

#include "Config.h"
#include "Type.h"
#include "MotionModel.h"
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
    };

    class SingleBotLaser2DGrid {
    public:
        SingleBotLaser2DGrid(const Vector3 &bot_pose, const BotParam &param,
            const std::string &fname);

        real rayCast(const Vector3 &pose) const;

        SensorData scan() const;
        void botAction(Control action);

        // Get Function
        Eigen::MatrixXf getMap(){
            return m_imageMap;
        }

        Vector3 getPose(){
            return m_pose;
        }

        BotParam getParam(){
            return m_param;
        }

        std::vector<Vector3> getTraj(){
            return m_traj;
        }
    private:
        Eigen::MatrixXf m_imageMap;
        Vector3 m_pose;
        BotParam m_param;
        std::vector <Vector3> m_traj;
    };
}

#endif