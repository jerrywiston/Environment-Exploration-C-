#ifndef _GRIDSLAM_GRIDMAP_H_
#define _GRIDSLAM_GRIDMAP_H_

#include <Config.h>
#include <opencv2/opencv.hpp>
#include <map>
#include <cstring>

struct CmpVector2 {
    template <class T>
    bool operator ()(const T &lhs, const T &rhs) const {
        return lhs[0]<rhs[0] || (lhs[0] == rhs[0] && lhs[1] < rhs[1]);
    }
};

namespace gslam {
    struct MapParam {
        real lo_occ;
        real lo_free;
        real lo_max;
        real lo_min;
    };

    struct BoundingBox {
        Vector2i min;
        Vector2i max;
    };

    template <size_t N>
    class MegaGrid {
    public:
        MegaGrid() {
            for(int i=0; i<N*N; i++)
                m_data[i] = 0;
        }
        real &operator()(const Vector2i &xy)
        { return m_data[xy[1]*N+xy[0]]; }
        const real &operator()(const Vector2i &xy) const
        { return m_data[xy[1]*N+xy[0]]; }
    private:
        real m_data[N*N];
    };

    class GridMap {
    public:
        GridMap(const MapParam &param, real gsize=1.0_r);

        real getGridProb(const Vector2i &pos) const;
        real getCoordProb(const Vector2 &pos) const;

        // [xy1, xy2)
        cv::Mat getMapProb(const Vector2i &xy1, const Vector2i &xy2) const;
        // boundary
        cv::Mat getMapProb() const;
        void line(const Vector2 &xy1, const Vector2 &xy2);

        real gridSize() const
        { return m_gsize; }

        BoundingBox getBoundary() const
        {return m_boundary;}

        size_t getMemoryUsage(){
            return m_mmap.size() * sizeof(GridMap);
        }
    private:
        MapParam m_param;
        real m_gsize;
        std::map<Vector2i, MegaGrid<MEGAGRID_SIZE>, CmpVector2> m_mmap;
        //std::map<Vector2i, real, CmpVector2> m_map;
        BoundingBox m_boundary;
        
    };
}

# endif