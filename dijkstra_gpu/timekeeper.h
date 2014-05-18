#pragma once
// STD
#include <chrono>
#include <functional>

class timekeeper
{
public:
    timekeeper() {}

    void start()
    {
        m_start = std::chrono::steady_clock::now();
    }

    void stop()
    {
        m_end = std::chrono::steady_clock::now();
    }

    long double duration()
    {
        return std::chrono::duration<long double, std::milli>(m_end - m_start).count();
    }

    long double measure_time(std::function<void()> fun)
    {
        start();
        fun();
        stop();
        return duration();
    }

private:
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_end;

private:
    timekeeper(const timekeeper&);
};