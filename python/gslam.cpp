#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <fmt/format.h>
#include <tuple>

#include <iostream>
#include <Type.h>

#include "GridMap.h"
#include "SingleBotLaser2D.h"
#include "ParticleFilter.h"

namespace py = pybind11;

static std::string toString(const gslam::Vector2 &v){

    return fmt::format("Vector2([{}, {}])", v[0], v[1]);
}

static gslam::Vector2 Vector2(std::vector<gslam::real> &v) {
    return {v[0], v[1]};
}

#define RECAST_BINARY_OPERATOR(T1, T2, T3, op) \
[] (const T1 &lhs, const T2 &rhs) -> T3 { \
    return T3((lhs) op (rhs)); \
}

#define RECAST_REV_BINARY_OPERATOR(T1, T2, T3, op) \
[] (const T1 &lhs, const T2 &rhs) -> T3 { \
    return T3((rhs) op (lhs)); \
}


PYBIND11_MODULE(gslam, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: gslam
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("toVector2", &Vector2, R"pbdoc(
        Convert numpy array to Vector2
    )pbdoc");


    py::class_<gslam::GridMap>(m, "GridMap")
        .def("__init__", [](gslam::GridMap &instance, const std::vector<gslam::real> &params, gslam::real gsize) {
            if (params.size() != 4)
                throw std::runtime_error("params must be [locc, free, max, min].");
            new (&instance) gslam::GridMap({params[0], params[1], params[2], params[3]}, gsize);
        }, py::arg("params"), py::arg("gsize")=1.0_r)
        .def("getMapProb", [] (gslam::GridMap &instance, const std::tuple<int, int> &xy1, const std::tuple<int, int> &xy2) {
            
            auto mat = instance.getMapProb({std::get<0>(xy1), std::get<1>(xy1)}, {std::get<0>(xy2), std::get<1>(xy2)});
            py::capsule free_when_done(mat.data(), [](void *f) {
                delete[] f;
            });
            
            return py::array_t<gslam::real>(
                {mat.rows(), mat.cols()}, // shape
                {mat.cols()*sizeof(gslam::real), sizeof(gslam::real)}, // C-style contiguous strides for double
                mat.data(), // the data pointer
                free_when_done); // numpy array references this parent
        })
        .def("getWholeMapProb", [](gslam::GridMap &instance) {
            auto mat = instance.getMapProb();
            py::capsule free_when_done(mat.data(), [](void *f) {
                delete[] f;
            });
            
            return py::array_t<gslam::real>(
                {mat.rows(), mat.cols()}, // shape
                {mat.cols()*sizeof(gslam::real), sizeof(gslam::real)}, // C-style contiguous strides for double
                mat.data(), // the data pointer
                free_when_done); // numpy array references this parent
        })
        .def("line", [] (gslam::GridMap &instance, const std::tuple<gslam::real, gslam::real> &xy1, const std::tuple<gslam::real, gslam::real> &xy2) {
            instance.line({std::get<0>(xy1), std::get<1>(xy1)}, {std::get<0>(xy2), std::get<1>(xy2)});
        });

    py::class_<gslam::SingleBotLaser2DGrid>(m, "SingleBotLaser2DGrid")
        .def("__init__", [] (gslam::SingleBotLaser2DGrid &instance, 
            const std::tuple<gslam::real, gslam::real, gslam::real> &pose, 
            py::array_t<uint8_t> map, py::dict d) {
            gslam::BotParam param;
            param.sensor_size = d["sensor_size"].cast<int>();
            param.start_angle = d["start_angle"].cast<gslam::real>();
            param.end_angle = d["end_angle"].cast<gslam::real>();
            param.max_dist = d["max_dist"].cast<gslam::real>();
            param.velocity = d["velocity"].cast<gslam::real>();
            param.rotate_step = d["rotate_step"].cast<gslam::real>();

            py::buffer_info info = map.request();
            if(info.ndim != 2)
                throw std::runtime_error("Number of map dimensions is not 2");
            auto wrapped = gslam::Storage2D<uint8_t>::Wrap(info.shape[1], info.shape[0], reinterpret_cast<uint8_t *>(info.ptr));
            new (&instance) gslam::SingleBotLaser2DGrid({std::get<0>(pose), std::get<1>(pose), std::get<2>(pose)}, param, wrapped);
        })
        .def("scan", [] (gslam::SingleBotLaser2DGrid &instance) {
            auto scan = instance.scan();
            using namespace pybind11::literals;
            return py::dict("sensor_size"_a=scan.sensor_size,
                "start_angle"_a=scan.start_angle,
                "end_angle"_a=scan.end_angle,
                "max_dist"_a=scan.max_dist,
                "data"_a=scan.data);
        })
        .def("action", [] (gslam::SingleBotLaser2DGrid &instance, int action) {
            instance.botAction(static_cast<gslam::Control>(action));
        })
        .def_property_readonly("pose", [] (gslam::SingleBotLaser2DGrid &instance) {
            auto pose = instance.getPose();
            return std::make_tuple(pose[0], pose[1], pose[2]);
        });

    py::class_<gslam::ParticleFilter>(m, "ParticleFilter")
        .def("__init__", [] (gslam::ParticleFilter &instance,
            const std::tuple<gslam::real, gslam::real, gslam::real> &pose, 
            py::dict d, 
            const gslam::GridMap &saved_map, 
            const int size) {
            gslam::BotParam param;
            param.sensor_size = d["sensor_size"].cast<int>();
            param.start_angle = d["start_angle"].cast<gslam::real>();
            param.end_angle = d["end_angle"].cast<gslam::real>();
            param.max_dist = d["max_dist"].cast<gslam::real>();
            param.velocity = d["velocity"].cast<gslam::real>();
            param.rotate_step = d["rotate_step"].cast<gslam::real>();
            new (&instance) gslam::ParticleFilter({std::get<0>(pose), std::get<1>(pose), std::get<2>(pose)}, param, saved_map, size);
        })
        .def("feed", [] (gslam::ParticleFilter &instance, int action, py::dict d) {
            gslam::SensorData readings;
            readings.sensor_size = d["sensor_size"].cast<int>();
            readings.start_angle = d["start_angle"].cast<gslam::real>();
            readings.end_angle = d["end_angle"].cast<gslam::real>();
            readings.max_dist = d["max_dist"].cast<gslam::real>();
            readings.data = d["data"].cast<std::vector<gslam::real>>();
            return instance.feed(static_cast<gslam::Control>(action), readings);
        })
        .def("resampling", &gslam::ParticleFilter::resampling)
        .def("bestSampleId", &gslam::ParticleFilter::bestSampleId);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}