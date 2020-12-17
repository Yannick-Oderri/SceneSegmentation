//
// Created by ynki9 on 12/15/20.
//

#ifndef PROJECT_EDGE_OBSERVER_H
#define PROJECT_EDGE_OBSERVER_H

template<typename NotificationType>
class Observer {
public:
    Observer() = default;
    Observer(const Observer &) = delete;
    Observer(const Observer &&) = delete;
    Observer &operator=(const Observer &) = delete;
    Observer &operator=(const Observer &&) = delete;

    virtual ~Observer() = default;

    virtual void NotifyTermination() = 0;
    virtual void ClearData() = 0;
    virtual void NotifyData(const NotificationType &data) = 0;


};

#endif //PROJECT_EDGE_OBSERVER_H
