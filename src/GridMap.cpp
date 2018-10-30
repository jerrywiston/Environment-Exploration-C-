#include "GridMap.h"
#include "Utils.h"
#include <cmath>
#include <algorithm>

namespace gslam
{
    GridMap::GridMap(const MapParam &param, real gsize)
        : m_param(param), m_gsize(gsize), m_boundary({{9999, 9999}, {-9999, -9999}})
    {

    }
    real GridMap::getGridProb(const Vector2i &pos) const
    {
        auto it = m_map.find(pos);
        if(it != m_map.cend()) {
            real tmp =  it->second;
            return exp(tmp) / (1+exp(tmp));
        }
        return 0.5_r;
    }

    real GridMap::getCoordProb(const Vector2 &pos) const
    {
        Vector2i xy{std::round(pos[0]/m_gsize), std::round(pos[1]/m_gsize)};
        return getGridProb(xy);
    }

    cv::Mat GridMap::getMapProb(const Vector2i &xy1, const Vector2i &xy2) const
    {
        cv::Mat ret(cv::Size(xy2[0]-xy1[0], xy2[1]-xy1[1]), CV_REAL_C1);
        for(int y=xy1[1]; y<xy2[1]; y++) {
            for(int x=xy1[0]; x<xy2[0]; x++) {
                ret.at<real>(y-xy1[1], x-xy1[0]) = getGridProb({x, y});
            }
        }
        return ret;
    }

    void GridMap::line(const Vector2 &xy1_, const Vector2 &xy2_)
    {
        Vector2i xy1 = Vector2i{std::round(xy1_[0]/m_gsize), std::round(xy1_[1]/m_gsize)};
        Vector2i xy2 = Vector2i{std::round(xy2_[0]/m_gsize), std::round(xy2_[1]/m_gsize)};

        std::vector<Vector2i> rec;
        utils::Bresenham(rec, xy1, xy2);
        for(int i=0; i<rec.size(); i++) {
            real change = m_param.lo_occ;
            if(i>=rec.size()-3)
                change = m_param.lo_free;
            auto it = m_map.find(rec[i]);
            if(it != m_map.cend()) {
                it->second += change;
                if(it->second > m_param.lo_max)
                    it->second = m_param.lo_max;
                else if(it->second < m_param.lo_min)
                    it->second = m_param.lo_min;
            } else {
                m_map[rec[i]] = change;
                m_boundary.min[0] = std::min(rec[i][0], m_boundary.min[0]);
                m_boundary.min[1] = std::min(rec[i][1], m_boundary.min[1]);
                m_boundary.max[0] = std::max(rec[i][0], m_boundary.max[0]);
                m_boundary.max[1] = std::max(rec[i][1], m_boundary.max[1]);
            }
        }
    }

}
