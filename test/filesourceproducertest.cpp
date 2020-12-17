//
// Created by ynki9 on 12/16/20.
//

#include <thread>

#include "gtest/gtest.h"
#include "context/context_factory.h"

#include "filter/simple_image_producer.h"

class SimpleImageProducerTest: public ::testing::Test {
protected:
    void SetUp() override {
        /// Initialize Application context
        AppContextBuilder app_ctx_builder;
        app_ctx_builder.setViewPortDimensions(800, 640);
        app_ctx_builder.setWindowTitle("Edge App");
        app_ctx_builder.setResDir("../../data");
        app_ctx_builder.setOutDir("./results");
        app_ctx = app_ctx_builder.Build();
    }

    // void TearDown() override{}

    std::unique_ptr<AppContext> app_ctx;
};
TEST_F(SimpleImageProducerTest, notify) {
    SimpleImageProducerConfig config = {false, 1, 20};
    SimpleImageProducer producer(app_ctx->getResMgr(), config);
    ASSERT_EQ(producer.getCurrentConfig().imageID_, 1);
    ASSERT_EQ(producer.getCurrentConfig().updateRate_, 20);

    std::thread producer_thread(&SimpleImageProducer::start, &producer);
    sleep(1);

    config.imageID_ = 2;
    config.updateRate_ = 30;
    producer.NotifyData(config);
    sleep(1);
    ASSERT_EQ(producer.getCurrentConfig().imageID_, 2);
    ASSERT_EQ(producer.getCurrentConfig().updateRate_, 30);


    ASSERT_FALSE(producer.getTerminationStatus());
    producer.NotifyTermination();

    producer_thread.join();

    ASSERT_TRUE(producer.getTerminationStatus());
}