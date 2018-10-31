#include <iostream>
#include "Config.h"
#include "GridMap.h"
#include "Utils.h"
#include "SingleBotLaser2D.h"
#include "Type.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace gslam;

cv::Mat Map2Image(Eigen::MatrixXf &m){
    cv::Mat img((int)m.rows(), (int)m.cols(), CV_8UC3, cv::Scalar(0,0,0));
    for(int i=0; i<m.rows(); ++i)
        for(int j=0; j<m.cols(); ++j){
            uchar value = (uchar)(m(i,j)*255);
            img.at<cv::Vec3b>(i,j)[0] = value;
            img.at<cv::Vec3b>(i,j)[1] = value;
            img.at<cv::Vec3b>(i,j)[2] = value;
        }
    return img;
}

cv::Mat Draw(cv::Mat img, Vector2 bot_pos, real theta, BotParam bot_param, SensorData sdata){
    auto plist = utils::EndPoints(bot_pos, theta, sdata);

    for(int i=0; i<plist.size(); ++i)
        cv::line(img, 
            cv::Point(int(bot_pos[0]), int(bot_pos[1])), 
            cv::Point(int(plist[i][0]), int(plist[i][1])),
            cv::Scalar(255,0,0), 1);
    cv::circle(img,cv::Point(int(bot_pos[0]), int(bot_pos[1])), 3, cv::Scalar(0,0,255), -1);
    return img;
}

void SensorMapping(GridMap &m, Vector2 pos, real theta, SensorData sdata){
    auto plist = utils::EndPoints(pos, theta, sdata);
    for(int i=0; i<sdata.sensor_size; ++i){
        if(sdata.data[i] > sdata.max_dist-1 || sdata.data[i] < 1)
            continue;
        m.line({pos[0], pos[1]},{plist[i][0], plist[i][1]});
    }
}

int main(int argc, char *argv[]) {
    Vector2 bot_pos = {150.0, 100.0};
    real theta = (real)0.0;
    BotParam bot_param;
    bot_param.sensor_size = 300;
    bot_param.start_angle = -30;
    bot_param.end_angle = 210;
    bot_param.max_dist = 150;
    bot_param.velocity = 3;
    bot_param.rotate_step = 6;
    SingleBotLaser2DGrid env(bot_pos, theta, bot_param, "./bin/map_large.png");
    cv::namedWindow("view", cv::WINDOW_AUTOSIZE);

    // Map
    gslam::GridMap gmap({0.4_r, -0.4_r, 5.0_r, -5.0_r});
    SensorMapping(gmap, bot_pos, theta, env.scan());
    BoundingBox bb = gmap.getBoundary(); 
    auto map = gmap.getMapProb(bb.min, bb.max);
    gslam::utils::VisualizeGrid(map);

    // Initialize
    cv::Mat img = Map2Image(env.getMap());
    img = Draw(img, env.getPos(), env.getTheta(), env.getParam(), env.scan());
    cv::imshow( "view", img );

    while(true){
        //std::cout << gmap.getSize() << std::endl;
        auto k = cv::waitKey(1);
        Control action = Control::eNone;
        switch(k){
            case 'w':
                action = Control::eForward;
                break;
            case 's':
                action = Control::eBackward;
                break;
            case 'a':
                action = Control::eTurnLeft;
                break;
            case 'd':
                action = Control::eTurnRight;
                break;
            case 'm':
                auto map = gmap.getMapProb({0, 0}, {300, 300});
                gslam::utils::VisualizeGrid(map);
                break;
        }
        if(action != Control::eNone){
            env.botAction(action);
            cv::Mat img = Map2Image(env.getMap());
            img = Draw(img, env.getPos(), env.getTheta(), env.getParam(), env.scan());
            cv::imshow( "view", img );

            SensorMapping(gmap, env.getPos(), env.getTheta(), env.scan());
            BoundingBox bb = gmap.getBoundary(); 
            auto map = gmap.getMapProb(bb.min, bb.max);
            gslam::utils::VisualizeGrid(map);
        }
    }              
    return 0;
}