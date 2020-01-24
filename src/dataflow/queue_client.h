//
// Created by ynki9 on 12/27/19.
//

#ifndef PROJECT_EDGE_QUEUE_CLIENT_H
#define PROJECT_EDGE_QUEUE_CLIENT_H

#include <condition_variable>
#include <boost/log/trivial.hpp>

/**
 * Pipeline Pipe element used for managing thread communication
 * @tparam Data Data type used along the pipeline
 */
template<class Data>
class QueueClient
{
private:
    std::queue<Data> m_queue_;
    std::mutex m_mutex_;
    std::condition_variable m_cv_;
    std::string m_name_;
    bool m_is_closed_; // not utilized as yet

public:
    QueueClient():
            m_name_(""),
            m_is_closed_(false){}

    QueueClient(std::string name):
            m_name_(name),
            m_is_closed_(false){}


    void waitData() {
        std::unique_lock<std::mutex> lck(m_mutex_);
        while(m_queue_.empty()){
            m_cv_.wait(lck);
        }

#ifdef DEBUG_LOG
        BOOST_LOG_TRIVIAL(debug) << "Closing Pipe " << m_name_;
#endif
    }

    void push(const Data& data) {
        std::unique_lock<std::mutex> lck(m_mutex_);
        m_queue_.push(data);
        m_cv_.notify_all();
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

    int size(){
        return m_queue_.size();
    }
};

#endif //PROJECT_EDGE_QUEUE_CLIENT_H
