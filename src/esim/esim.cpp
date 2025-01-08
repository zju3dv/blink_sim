#include "esim.h"
#include "utils.h"

#include <iostream>
#include <memory>
#include <random>

void EventSimulator::_init_image(Image<float> img, Time time) {
    // std::cout << "Initialized event camera simulator with sensor size: "
    //           << img.size();
    is_initialized_ = true;
    last_img_ = img.clone();
    ref_values_ = img.clone();
    // last_event_timestamp_ = TimestampImage::zeros(img.size());
    last_event_timestamp_ = TimestampImage(img.size());
    current_time_ = time;
    size_ = img.size();
}

float EventSimulator::sampleNormalDistribution(float mean, float sigma) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(mean, sigma);
    return distribution(generator);
}

inline constexpr std::int64_t secToNanosec(float seconds) {
    return static_cast<std::int64_t>(seconds * 1e9);
}

inline constexpr float nanosecToSecTrunc(std::int64_t nanoseconds) {
    return static_cast<float>(nanoseconds) / 1e9;
}
py::array EventSimulator::generate_events(const py::array_t<float>& _img,
                                          Time time) {
    py::buffer_info buf = _img.request();
    Image<float> img(buf.shape[0], buf.shape[1], (float*)buf.ptr);

    CHECK_GE(time, 0);
    Image<float> preprocessed_img = img.clone();
    if (config_.use_log_image) {
        // std::cout << "Converting the image to log image with eps = "
        //                      << config_.log_eps << ".\n";
        // cv::log(config_.log_eps + img, preprocessed_img);
        preprocessed_img = preprocessed_img + config_.log_eps;
        preprocessed_img.log();
    }

    if (!is_initialized_) {
        _init_image(preprocessed_img, time);
        return {};
    }

    // For each pixel, check if new events need to be generated since the last
    // image sample
    static constexpr float tolerance = 1e-6;
    Events events;
    Duration delta_t_ns = time - current_time_;

    CHECK_GT(delta_t_ns, 0u);
    CHECK_EQ(img.size(), size_);

    for (int y = 0; y < size_.height; ++y) {
        for (int x = 0; x < size_.width; ++x) {
            float itdt = preprocessed_img(y, x);
            float it = last_img_(y, x);
            float prev_cross = ref_values_(y, x);

            if (std::fabs(it - itdt) > tolerance) {
                float pol = (itdt >= it) ? +1.0 : -1.0;
                float C = (pol > 0) ? config_.Cp : config_.Cm;
                float sigma_C =
                    (pol > 0) ? config_.sigma_Cp : config_.sigma_Cm;
                if (sigma_C > 0) {
                    C += sampleNormalDistribution(0, sigma_C);
                    constexpr float minimum_contrast_threshold = 0.01;
                    C = std::max(minimum_contrast_threshold, C);
                }
                float curr_cross = prev_cross;
                bool all_crossings = false;

                do {
                    curr_cross += pol * C;

                    if ((pol > 0 && curr_cross > it && curr_cross <= itdt) ||
                        (pol < 0 && curr_cross < it && curr_cross >= itdt)) {
                        Duration edt =
                            (curr_cross - it) * delta_t_ns / (itdt - it);
                        Time t = current_time_ + edt;

                        // check that pixel (x,y) is not currently in a
                        // "refractory" state i.e. |t-that last_timestamp(x,y)|
                        // >= refractory_period
                        const Time last_stamp_at_xy =
                            secToNanosec(last_event_timestamp_(y, x));
                        CHECK_GE(t, last_stamp_at_xy);
                        const Duration dt = t - last_stamp_at_xy;
                        if (last_event_timestamp_(y, x) == 0 ||
                            dt >= config_.refractory_period_ns) {
                            events.push_back(Event(x, y, t, pol > 0));
                            last_event_timestamp_(y, x) = nanosecToSecTrunc(t);
                        } else {
                            std::cout
                                << "Dropping event because time since last "
                                   "event  ("
                                << dt << " ns) < refractory period ("
                                << config_.refractory_period_ns << " ns).\n";
                        }
                        ref_values_(y, x) = curr_cross;
                    } else {
                        all_crossings = true;
                    }
                } while (!all_crossings);
            }   // end tolerance
        }       // end for each pixel
    }

    // update simvars for next loop
    current_time_ = time;
    last_img_ = preprocessed_img.clone();   // it is now the latest image

    // Sort the events by increasing timestamps, since this is what
    // most event processing algorithms expect
    sort(events.begin(), events.end(),
         [](const Event& a, const Event& b) -> bool { return a.t < b.t; });

    std::vector<float> data(events.size() * 4);
    for (int i = 0; i < events.size(); i++) {
        data[i * 4 + 0] = events[i].t;
        data[i * 4 + 1] = events[i].y;
        data[i * 4 + 2] = events[i].x;
        data[i * 4 + 3] = events[i].pol;
    }

    auto pyarray = as_pyarray(std::move(data));
    pyarray.resize({static_cast<int>(events.size()), 4});

    return pyarray;
}

// wrap as Python module
PYBIND11_MODULE(esim, m) {
    py::class_<EventSimulator>(m, "EventSimulator")
        .def(py::init<>())
        .def("generate_events", &EventSimulator::generate_events,
             "generate_events");
}
