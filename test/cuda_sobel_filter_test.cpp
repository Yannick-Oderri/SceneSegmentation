//
// Created by ynki9 on 12/30/19.
//
#include <thread>

#include "gtest/gtest.h"
#include "cuda_sobel_filter.h"
#include "kinect_bridge_producer.h"
#include "simple_image_producer.h"
#include "simple_viewer_consumer.h"
#include "gl_edge_disc_filter.h"

TEST(CudaSobelTests, SimpleImageProducerTest) {
    AppContext::Builder builder;
    builder.initializeCuda();
    AppContext* const appContext = builder.Build();

#ifdef WITH_KINECTV2
    FreenectPipeProducer producer;
//    producer.initialize();
    std::thread producer_tread(&FreenectPipeProducer::start, &producer);
#else
    SimpleImageProducer producer;
    producer.initialize();
    std::thread producer_tread(&SimpleImageProducer::start, &producer);
#endif
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    GLEdgeDiscFilter edge_filter(producer.getOutQueue());
    edge_filter.initialize();

    SimpleViewerConsumer viewer_consumer(edge_filter.getOutQueue());

    std::thread sobel_thread(&GLEdgeDiscFilter::start, &edge_filter);
    std::thread viewer_thread(&SimpleViewerConsumer::start, &viewer_consumer);

    producer_tread.join();
    sobel_thread.join();
    viewer_thread.join();
}

TEST(DISABLED_CudaSobelTests, SimpleImageProducerTest) {
    AppContext::Builder builder;
    builder.initializeCuda();
    AppContext* const appContext = builder.Build();

    SimpleImageProducer producer;
    producer.initialize();

    CudaSobelFilter sobel_filter(producer.getOutQueue());
    sobel_filter.intialize(appContext->getCudaDevice());

    SimpleViewerConsumer viewer_consumer(sobel_filter.getOutQueue());


    std::thread producer_tread(&SimpleImageProducer::start, &producer);
    std::thread sobel_thread(&CudaSobelFilter::start, &sobel_filter);
    std::thread viewer_thread(&SimpleViewerConsumer::start, &viewer_consumer);

    producer_tread.join();
    sobel_thread.join();
    viewer_thread.join();
}

TEST(DISABLED_CudaSobelTests, PipelineIntegration) {
    AppContext::Builder builder;
    builder.initializeCuda();
    AppContext* const appContext = builder.Build();

    FreenectPipeProducer producer;
    producer.initializeFreenectContext();

    CudaSobelFilter sobel_filter(producer.getOutQueue());
    sobel_filter.intialize(appContext->getCudaDevice());

    SimpleViewerConsumer viewer_consumer(sobel_filter.getOutQueue());


    std::thread producer_tread(&FreenectPipeProducer::start, &producer);
    std::thread sobel_thread(&CudaSobelFilter::start, &sobel_filter);
    std::thread viewer_thread(&SimpleViewerConsumer::start, &viewer_consumer);

    producer_tread.join();
    sobel_thread.join();
    viewer_thread.join();
}