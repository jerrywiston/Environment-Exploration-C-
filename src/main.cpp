#include <iostream>
#include "Config.h"
#include "GridMap.h"
#include "Utils.h"
#include "SingleBotLaser2D.h"
#include "Type.h"
#include "ParticleFilter.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>

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

cv::Mat Draw(cv::Mat img, Vector3 bot_pose, BotParam bot_param, SensorData sdata){
    auto plist = utils::EndPoints(bot_pose, sdata);

    for(int i=0; i<plist.size(); ++i)
        cv::line(img, 
            cv::Point(int(bot_pose[0]), int(bot_pose[1])), 
            cv::Point(int(plist[i][0]), int(plist[i][1])),
            cv::Scalar(0,255,0), 1);
    cv::circle(img,cv::Point(int(bot_pose[0]), int(bot_pose[1])), 3, cv::Scalar(0,0,255), -1);
    return img;
}

cv::Mat DrawParticle(cv::Mat img, ParticleFilter pf){
    for(int i=0; i<pf.getSize(); ++i){
        Vector3 pose = pf.getPose(i);
        cv::circle(img,cv::Point(int(pose[0]), int(pose[1])), 1, cv::Scalar(255,0,0), -1);
    }
    return img;
}

void SensorMapping(GridMap &m, Vector3 &pose, SensorData sdata){
    auto plist = utils::EndPoints(pose, sdata);
    for(int i=0; i<sdata.sensor_size; ++i){
        if(sdata.data[i] > sdata.max_dist-1 || sdata.data[i] < 1)
            continue;
        m.line({pose[0], pose[1]},{plist[i][0], plist[i][1]});
    }
}

int main(int argc, char *argv[]) {
    Vector3 bot_pose = {120.0, 80.0, 180.0};
    BotParam bot_param;
    bot_param.sensor_size = 240;
    bot_param.start_angle = -30;
    bot_param.end_angle = 210;
    bot_param.max_dist = 130;
    bot_param.velocity = 3;
    bot_param.rotate_step = 6;
    SingleBotLaser2DGrid env(bot_pose, bot_param, "./bin/map2.png");
    cv::namedWindow("view", cv::WINDOW_AUTOSIZE);

    // Map
    gslam::GridMap gmap({-0.4_r, 0.4_r, 5.0_r, -5.0_r});
    SensorMapping(gmap, bot_pose, env.scan());
    BoundingBox bb = gmap.getBoundary(); 
    auto map = gmap.getMapProb(bb.min, bb.max);
    gslam::utils::VisualizeGrid(map, "map_env");

    // Particle Filter
    ParticleFilter pf(bot_pose, bot_param, gmap, 100);

    // Initialize
    cv::Mat img = Map2Image(env.getMap());
    img = Draw(img, env.getPose(), env.getParam(), env.scan());
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
        }
        if(action != Control::eNone){
            env.botAction(action);

            //particle filter
            real Neff = pf.feed(action, env.scan());
            std::cout << "Neff: " << Neff << std::endl;
            if(Neff < 0.5){
                std::cout << "Resampling ..." << std::endl;
                pf.resampling();
                std::cout << "Done !!" << std::endl;
            }
            

            cv::Mat img = Map2Image(env.getMap());
            img = Draw(img, env.getPose(), env.getParam(), env.scan());
            img = DrawParticle(img, pf);
            cv::imshow( "view", img );

            SensorMapping(gmap, env.getPose(), env.scan());
            BoundingBox bb = gmap.getBoundary(); 
            auto map = gmap.getMapProb(bb.min, bb.max);
            gslam::utils::VisualizeGrid(map, "map_env");

            GridMap gmap_p = pf.getParticle(0).getMap();
            BoundingBox bb2 = gmap_p.getBoundary();
            auto map2 = gmap_p.getMapProb(bb2.min, bb2.max);
            gslam::utils::VisualizeGrid(map2, "map_particle");
        }
    }              
    return 0;
}