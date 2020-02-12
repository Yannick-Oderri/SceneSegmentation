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
#include "contour_processor.h"

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

    ContourProcessorPipeFilter contour_processor(edge_filter.getOutQueue(), new LineSegmentContourPolicy());
    contour_processor.initialize();

    std::thread contour_detector_thread(&GLEdgeDiscFilter::start, &edge_filter);
    std::thread contour_processor_thread(&ContourProcessorPipeFilter::start, &contour_processor);

    producer_tread.join();
    contour_detector_thread.join();
    contour_processor_thread.join();
}
