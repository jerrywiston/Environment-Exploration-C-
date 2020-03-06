#ifndef _GRIDSLAM_TYPE_H_
#define _GRIDSLAM_TYPE_H_

#include "Config.h"
#include <vector>
#include <cstdint>

namespace gslam {
    struct BotParam {
        int sensor_size;
        // in degree
        real start_angle;
        real end_angle;
        real max_dist;

        //motion model
        real noise_nor;
        real noise_tan;
        real noise_ang;
    };

    struct SensorData {
        int sensor_size;
        real start_angle;
        real end_angle;
        real max_dist;
        std::vector<real> data;
    };

    union rgb_t {
        struct {
            uint8_t r,g,b;
        };
        uint8_t data[3];
    };

    template <class T>
    class Storage2D {
    public:
        Storage2D(const Storage2D<T> &rhs);
        Storage2D(uint32_t width, uint32_t height);
        const T *data() const
        { return m_data; }

        static Storage2D<T> Wrap(uint32_t width, uint32_t height, T *data);
        uint32_t width() const
        { return m_width; }
        uint32_t height() const
        { return m_height; }
        uint32_t cols() const
        { return m_width; }
        uint32_t rows() const
        { return m_height; }
    private:
        Storage2D()=default;
        uint32_t m_width;
        uint32_t m_height;
        T *m_data;
    };
    template <class T>
    Storage2D<T>::Storage2D(const Storage2D<T> &rhs)
        : m_width(rhs.m_width), m_height(rhs.m_height), m_data(rhs.m_data)
    {

    }
    template <class T>
    Storage2D<T>::Storage2D(uint32_t width, uint32_t height)
        : m_width(width), m_height(height), m_data(new T[width*height])
    {

    }
    template <class T>
    Storage2D<T> Storage2D<T>::Wrap(uint32_t width, uint32_t height, T *data)
    {
        Storage2D<T> ret;
        ret.m_width = width;
        ret.m_height = height;
        ret.m_data = data;
        return ret;
    }
}

#endif