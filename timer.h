#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <iostream>
#include <string>


struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
};

// 开始计时
inline void startTime(Timer* t) {
    t->start = std::chrono::high_resolution_clock::now();
}

// 停止计时
inline void stopTime(Timer* t) {
    t->end = std::chrono::high_resolution_clock::now();
}

// 打印耗时，单位毫秒
inline void printElapsedTime(const Timer& t, const std::string& label = "Elapsed time") {
    std::chrono::duration<double, std::milli> elapsed = t.end - t.start;
    std::cout << label << ": " << elapsed.count() << " ms" << std::endl;
}



#endif // TIMER_H
