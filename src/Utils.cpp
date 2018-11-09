#include "Utils.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>



namespace gslam
{
    void swap(real &a, real &b){
        real c = a;
        a = b;
        b = c;
    }
    
namespace utils
{
    void VisualizeGrid(const cv::Mat &mat, const std::string name)
    {
        assert(mat.type() == CV_REAL_C1);
        cv::Mat target;
        cv::Mat m = mat*255.0_r;
        target.convertTo(m, CV_8UC1);
        cv::imshow(name, mat);
    }
    //void Line( const float x1, const float y1, const float x2, const float y2)
    void Bresenham(std::vector<Vector2i> &rec, const Vector2i &xy1, const Vector2i &xy2)
    {
        rec.clear();
        real x1 = (real)xy1[0];
        real y1 = (real)xy1[1];
        real x2 = (real)xy2[0];
        real y2 = (real)xy2[1];

        // Bresenham's line algorithm
        const bool steep = (fabs(y2 - y1) > fabs(x2 - x1));
        if(steep)
        {
            swap(x1, y1);
            swap(x2, y2);
        }
    
        if(x1 > x2)
        {
            swap(x1, x2);
            swap(y1, y2);
        }
    
        const real dx = x2 - x1;
        const real dy = fabs(y2 - y1);
    
        real error = dx / 2.0f;
        const int ystep = (y1 < y2) ? 1 : -1;
        int y = (int)y1;
    
        const int maxX = (int)x2;
    
        for(int x=(int)x1; x<maxX; x++)
        {
            if(steep)
                rec.push_back({y,x});
            else
                rec.push_back({x,y});
    
            error -= dy;
            if(error < 0)
            {
                y += ystep;
                error += dx;
            }
        }

        int tmp1 = std::pow(rec[0][0]-xy1[0], 2) + std::pow(rec[0][1]-xy1[1], 2);
        int tmp2 = std::pow(rec[rec.size()-1][0]-xy1[0], 2) + std::pow(rec[rec.size()-1][1]-xy1[1], 2);
        if(tmp1 > tmp2)
            std::reverse(rec.begin(),rec.end());

    }

    real Gaussian(real mu, real sigma)
    {
        thread_local std::default_random_engine generator;
        std::normal_distribution<real> dist(mu, sigma);
        return dist(generator);
    }

    real GaussianPDF(real x, real mu, real sigma)
    {
        return 1._r/(std::sqrt(2*3.1415926_r)*sigma)*std::exp(-std::pow((x-mu)/sigma, 2)/2);
    }

    std::vector<Vector2> EndPoints(Vector3 pose, SensorData sdata)
    {
        real inter = (sdata.end_angle - sdata.start_angle) / (sdata.sensor_size-1);
        std::vector<Vector2> endList(sdata.sensor_size);
        for(int i=0; i<sdata.sensor_size; ++i){
            real angle = pose[2] + sdata.start_angle + i*inter;
            Vector2 tmp = { pose[0]+sdata.data[i]*cos(Deg2Rad(angle-90)),
                            pose[1]+sdata.data[i]*sin(Deg2Rad(angle-90))};
            //#include <iostream>
            //std::cout<< pos[1] << " " << tmp << std::endl << std::endl;
            endList[i] = tmp;
        }
        return endList;
    }

    real Deg2Rad(real deg){
        return deg*3.1415926_r/180.0_r;
    }
}
}