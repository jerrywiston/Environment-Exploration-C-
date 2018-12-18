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
        const real half = (MEGAGRID_SIZE-1) / 2;
		Vector2i gridCoord = { std::ceil((std::abs(pos[0]) - half) / MEGAGRID_SIZE),
			std::ceil((std::abs(pos[1]) - half) / MEGAGRID_SIZE) };
		gridCoord[0] *= pos[0] < 0? -1:1;
		gridCoord[1] *= pos[1] < 0 ? -1 : 1;
        auto it = m_mmap.find(gridCoord);
        if(it != m_mmap.cend()) {
            Vector2i girdLeftTop = gridCoord * MEGAGRID_SIZE - Vector2i{half, half};
            real tmp = it->second(pos-girdLeftTop);
            return std::exp(tmp) / (1+std::exp(tmp));
        }
        return 0.5_r;
    }

    real GridMap::getCoordProb(const Vector2 &pos) const
    {
        Vector2i xy{std::round(pos[0]/m_gsize), std::round(pos[1]/m_gsize)};
        return getGridProb(xy);
    }

#ifdef WITH_OPENCV
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
#else
    Storage2D<real> GridMap::getObserv(const Vector2i &xy, real theta, int lx, int ly) const
    {
        assert(lx > 0 && ly > 0);
        Vector2i xy_ = {std::round(xy[0]/m_gsize), std::round(xy[1]/m_gsize)};
        real rad = utils::Deg2Rad(theta + 90._r);
        real *data = new real[lx*2 * ly*2];
        int c = 0;
        for(int i=-ly; i<ly; i++) {
            for(int j=-lx; j<lx; j++) {
                Vector2i sp = {std::round(xy[0]+j*std::cos(rad)-i*std::sin(rad)), std::round(xy[1]+j*std::sin(rad)+i*std::cos(rad))};
                data[c++] = getGridProb(sp);
            }
        }
        return Storage2D<real>::Wrap(lx*2, ly*2, data);
    }

    Storage2D<real> GridMap::getMapProb(const Vector2i &xy1, const Vector2i &xy2) const
    {
        real *data = new real[(xy2[0]-xy1[0])*(xy2[1]-xy1[1])];
        auto ret = Storage2D<real>::Wrap(xy2[0]-xy1[0], xy2[1]-xy1[1], data);
        for(int y=xy1[1]; y<xy2[1]; y++) {
            for(int x=xy1[0]; x<xy2[0]; x++) {
                *(data++) = getGridProb({x, y});
            }
        }
        return ret;
    }
#endif
    real GridMap::line(const Vector2 &xy1_, const Vector2 &xy2_)
    {
        real delta_info = 0.0_r;
        Vector2i xy1 = Vector2i{std::round(xy1_[0]/m_gsize), std::round(xy1_[1]/m_gsize)};
        Vector2i xy2 = Vector2i{std::round(xy2_[0]/m_gsize), std::round(xy2_[1]/m_gsize)};

        std::vector<Vector2i> rec;
        utils::Bresenham(rec, xy1, xy2);

        const real half = (MEGAGRID_SIZE-1) / 2;

        for(int i=0; i<rec.size(); i++) {
            real change = m_param.lo_occ;
            if(i>=rec.size()-3)
                change = m_param.lo_free;
            //std::cout << ">>>" << rec[i] << "\n";
			Vector2i gridCoord = { std::ceil ((std::abs(rec[i][0]) - half) / MEGAGRID_SIZE),
				std::ceil((std::abs(rec[i][1]) - half) / MEGAGRID_SIZE )};
			gridCoord[0] *= rec[i][0] < 0 ? -1 : 1;
			gridCoord[1] *= rec[i][1] < 0 ? -1 : 1;
            //std::cout << ">" << gridCoord << "\n";
            Vector2i girdLeftTop = gridCoord * MEGAGRID_SIZE - Vector2i{half, half};
			Vector2i pp = rec[i] - girdLeftTop;
            auto it = m_mmap.find(gridCoord);
            real old_grid_p = 0.5_r;
            real new_grid_p = 0;
            if(it != m_mmap.cend()) {
                auto &grid_val = it->second(rec[i]-girdLeftTop);
                old_grid_p = std::exp(grid_val)/(1+std::exp(grid_val));
                grid_val += change;
                if(grid_val > m_param.lo_max)
                    grid_val = m_param.lo_max;
                else if(grid_val < m_param.lo_min)
                    grid_val = m_param.lo_min;
                new_grid_p = std::exp(grid_val)/(1+std::exp(grid_val));
            } else {
                
                auto &newmega = m_mmap[gridCoord];
                newmega(pp) = change;
                new_grid_p = std::exp(change)/(1+std::exp(change));
            }
            delta_info += -new_grid_p*std::log(new_grid_p) - (-old_grid_p*std::log(old_grid_p));
            m_boundary.min[0] = std::min(rec[i][0], m_boundary.min[0]);
            m_boundary.min[1] = std::min(rec[i][1], m_boundary.min[1]);
            m_boundary.max[0] = std::max(rec[i][0], m_boundary.max[0]);
            m_boundary.max[1] = std::max(rec[i][1], m_boundary.max[1]);
        }
        return delta_info;
    }
    

}
