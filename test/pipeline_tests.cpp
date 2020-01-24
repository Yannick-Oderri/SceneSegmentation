//
// Created by ynki9 on 12/27/19.
//

#include <string>
#include <algorithm>
#include <thread>
#include <chrono>
#include <stdio.h>
#include "gtest/gtest.h"

#include "pipeline_filter.h"
#include "pipeline.h"

class StringProducer: public ProducerPipeFilter<std::string> {
public:
    StringProducer(QueueClient<std::string>* out_queue):
            ProducerPipeFilter(out_queue) {}

    int count_ = 0;
    virtual void start() {
        while(true){
            this->out_queue_->push(std::to_string(count_));
            count_++;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "" << count_ << std::endl;
        }
    }
};

class StringFilter: public PipeFilter<std::string, std::string> {
public:
    StringFilter(QueueClient<std::string>* in_queue, QueueClient<std::string>* out_queue):
            PipeFilter(in_queue, out_queue) {}
    virtual void start() {
        while(true) {
            this->in_queue_->waitData();
            std::string val = this->in_queue_->front();
            std::string tval = val + ": Filter Level";
            std::transform(tval.begin(), tval.end(), tval.begin(), ::toupper);
            this->in_queue_->pop();
            this->out_queue_->push(tval);
        }
    }
};

class StringViewer: public PipeFilter<std::string, std::string> {
public:
    StringViewer(QueueClient<std::string>* in_queue, QueueClient<std::string>* out_queue):
            PipeFilter(in_queue, out_queue){}

    virtual void start() {
        while(true){
            this->in_queue_->waitData();
            std::string val = this->in_queue_->front();
            std::cout << val << std::endl;

            this->in_queue_->pop();
        }
    }
};

TEST(PipelineTest, integration_test) {
    auto* producer_pipe = new QueueClient<std::string>();
    auto* filter_pipe = new QueueClient<std::string>();
    auto* consumer_pipe = new QueueClient<std::string>();

    StringProducer producer(producer_pipe);
    std::thread producer_thread(&StringProducer::start, &producer);

    StringFilter filter(producer_pipe, filter_pipe);
    std::thread filter_thread(&StringFilter::start, &filter);

    StringViewer consumer(filter_pipe, consumer_pipe);
    std::thread consumer_thread(&StringViewer::start, &consumer);

    producer_thread.join();
    filter_thread.join();
    consumer_thread.join();
}

TEST(PipelineTest, QueueClient_Push) {
    // Arrange
    QueueClient<string> qclient;

    EXPECT_EQ(qclient.size(), 0);
    EXPECT_TRUE(qclient.empty());

    qclient.push("Dogs and cats");
    EXPECT_EQ(qclient, 1);
    EXPECT_EQ(qclient.front(), "Dogs and Cats");
    EXPECT_FALSE(qclient.empty());

    qclient.pop();
    EXPECT_TRUE(qclient.empty());
    EXPECT_EQ(qclient.size(), 0);

}