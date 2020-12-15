//
// Created by ynki9 on 12/15/20.
//

#ifndef PROJECT_EDGE_FRAME_OBSERVER_H
#define PROJECT_EDGE_FRAME_OBSERVER_H

#include <memory>

template<typename NotificationType>
class FrameObserver {
public:
    virtual void FrameObserver(const NotificationType &data) = 0;
    virtual void NotifyTermination() = 0;
    virtual void ClearData() = 0;
    virtual void NotifyData() = 0;

    virtual void ~FrameObserver() = default;

    FrameObserver() = default;
};


#endif //PROJECT_EDGE_FRAME_OBSERVER_H
