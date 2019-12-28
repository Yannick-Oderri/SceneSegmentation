//
// Created by ynki9 on 12/27/19.
//

#ifndef PROJECT_EDGE_QUEUE_CLIENT_H
#define PROJECT_EDGE_QUEUE_CLIENT_H

#include <condition_variable>

template<class Data>
class QueueClient
{
private:
    std::queue<Data> m_queue_;
    std::mutex m_mutex_;
    std::condition_variable m_cv_;
public:
    void waitData() {
        std::unique_lock<std::mutex> lck(m_mutex_);
        while(m_queue_.empty()){
            m_cv_.wait(lck);
        }
    }

    void push(const Data& data) {
        std::unique_lock<std::mutex> lck(m_mutex_);
        m_queue_.push(data);
    }

    bool empty() const {
        std::unique_lock<std::mutex> lck(m_mutex_);
        m_queue_.empty();
    }

    Data& front() {
        std::unique_lock<std::mutex> lck(m_mutex_);
        return m_queue_.front();
    }

    Data const& front() const {
        std::unique_lock<std::mutex> lck(m_mutex_);
        return m_queue_.front();
    }

    void pop() {
        std::unique_lock<std::mutex> lck(m_mutex_);
        m_queue_.pop();
    }
};

#endif //PROJECT_EDGE_QUEUE_CLIENT_H
