#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

/*
 * The EventSimulator takes as input a sequence of stamped images,
 * assumed to be sampled at a "sufficiently high" framerate,
 * and simulates the principle of operation of an idea event camera
 * with a constant contrast threshold C.
 * Pixel-wise intensity values are linearly interpolated in time.
 *
 * The pixel-wise voltages are reset with the values from the first image
 * which is passed to the simulator.
 */

namespace py = pybind11;

using Time = uint32_t;
using Duration = uint32_t;

struct Event {
    Event(uint16_t x, uint16_t y, Time t, bool pol)
        : x(x), y(y), t(t), pol(pol) {}

    uint16_t x;
    uint16_t y;
    Time t;
    bool pol;
};

using Events = std::vector<Event>;

class Size {
public:
    Size() = default;
    Size(int _h, int _w) : height(_h), width(_w) {}
    // std::ostream& operator<<(std::ostream& os, const Size& s) {
    //     os << "(" << s.height << ", " << s.width << ")\n";
    //     return os;
    // }
    int height, width;
    bool operator==(Size s) {
        return height == s.height && width == s.width;
    }
};

template <class T>
class Image {
   public:
    Image() = default;
    Image(int _height, int _width) {
        size_ = Size(_height, _width);
        data.resize(_height * _width);
        for (int i = 0; i < data.size(); i++) {
            data[i] = 0;
        }
    }
    Image(Size size) {
        size_ = size;
        data.resize(size.height * size.width);
        for (int i = 0; i < data.size(); i++) {
            data[i] = 0;
        }
    }

    Image(int _height, int _width, T* ptr) {
        size_ = Size(_height, _width);
        data.resize(_height * _width);
        data.assign(ptr, ptr + _height * _width);
    }
    T operator()(int y, int x) const { return data[y * size_.width + x]; }
    T& operator()(int y, int x) { return data[y * size_.width + x]; }
    Image clone() {
        Image<T> img;
        img.size_ = Size(size_.height, size_.width);
        img.data = data;
        return img;
    }
    void log() {
        for (int i = 0; i < data.size(); i++) {
            data[i] = std::log(data[i]);
        }
    }
    Image& operator+(T v) {
        for (int i = 0; i < data.size(); i++) {
            data[i] += v;
        }
        return *this;
    }
    Size size() { return size_; }

   private:
    std::vector<T> data;
    Size size_;
};

class EventSimulator {
   public:
    struct Config {
        float Cp;
        float Cm;
        float sigma_Cp;
        float sigma_Cm;
        Duration refractory_period_ns;
        bool use_log_image;
        float log_eps;
    };

    using TimestampImage = Image<float>;

    EventSimulator() : is_initialized_(false), current_time_(0) {
        config_.Cp = 0.18;
        config_.Cm = 0.18;
        config_.sigma_Cp = 0.025;
        config_.sigma_Cm = 0.025;
        config_.refractory_period_ns = 0;
        config_.use_log_image = true;
        config_.log_eps = 0.001;

        std::cout << "Contrast thresholds: C+ = " << config_.Cp
                  << " , C- = " << config_.Cm << "\n";
    }

    py::array generate_events(const py::array_t<float>& img, Time time);

   private:
    bool is_initialized_;
    Time current_time_;
    Image<float> ref_values_;
    Image<float> last_img_;
    TimestampImage last_event_timestamp_;
    Size size_;

    Config config_;

    float sampleNormalDistribution(float mean, float sigma);
    void _init_image(Image<float> img, Time time);
};
