//
// Created by ynki9 on 12/29/19.
//

#include <thread>
#include "gtest/gtest.h"
#include "kinect_bridge_producer.h"
#include "simple_viewer_consumer.h"


TEST(DISABLED_FreenectTest, InitializeFreenect) {
    FreenectPipeProducer producer;
    producer.initializeFreenectContext();

    ASSERT_NE(producer.getFreenectDeviceID(), nullptr);
    ASSERT_EQ(producer.isInitialized(), true);

    producer.stopStream();
    producer.closeStream();
}

TEST(FreenectTest, ConsumeImage) {
    FreenectPipeProducer producer;
    producer.initializeFreenectContext();

    SimpleViewerConsumer consumer(producer.getOutQueue());

    std::thread producer_thread(&FreenectPipeProducer::start, &producer);
    std::thread consumer_thread(&SimpleViewerConsumer::start, &consumer);


    producer_thread.join();
    consumer_thread.join();

}