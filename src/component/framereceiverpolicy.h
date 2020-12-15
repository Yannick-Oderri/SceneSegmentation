//
// Created by ynki9 on 12/15/20.
//

#ifndef PROJECT_EDGE_FRAMERECEIVERPOLICY_H
#define PROJECT_EDGE_FRAMERECEIVERPOLICY_H

// System Headers
#include <mutex>
#include <list>

// Project Headers
#include "component/pipeline_policy.h"
#include "dataflow/frame_observer.h"

template<typename T>
class FrameReceiverPolicy : public PipelinePolicy {
public:
    FrameReceiverPolicy() = default;
    ~FrameReceiverPolicy(){
        NotifyTermination();
    }

    bool executePolicy(){

    }

    void RegisterObserver(std::shared_ptr<FrameObserver<T>> &&observer){
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_primed){
            observer->NotifyData();
        }
        m_observers.emplace_back(std::move(observer));
    }

    void NotifyObservers(const T &data){
        std::lock_guard<std::mutex>(m_mutex);

        m_primed = true;
        for (auto wpObserver = m_observers.begin(); wpObserver != m_observers.end();){
            auto spObserver = wpObserver->lock();
            if (spObserver) {
                spObserver->NotifyData(data);
                ++wpObserver;
            }else{
                auto toDelete = wpObserver;
                ++wpObserver;
                m_observers.erase(toDelete);
            }
        }

    }

    void NotifyTermination(Observer& observer){

    }
private:
    std::list<std::weak_ptr<FrameObserver<T>>> m_observers;
    std::mutex m_mutex;
    bool m_primed = false;

};


#endif //PROJECT_EDGE_FRAMERECEIVERPOLICY_H
