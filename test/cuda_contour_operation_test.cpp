//
// Created by ynki9 on 4/1/20.
//
#include <string>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "component/cu_contour_opr.h"
#include "filter/simple_image_producer.h"
#include "component/contour_policy.h"
#include "component/depth_img_policy.h"
#include <opencv2/opencv.hpp>

using ::testing::Return;


class MockResMgr: public ResMgr{
public:
    MOCK_METHOD(cv::Mat, loadColorImage, (std::string, int), (override));
    MOCK_METHOD(cv::Mat, loadDepthImage, (std::string, int), (override));
};

TEST (DISABLED_cudaContourOperationUnitTest, ContourOperation) {
    MockResMgr resMgr;
    cv::Mat dimg = cv::imread("../../data/images/depth/test0.png", cv::IMREAD_UNCHANGED);
    cv::Mat cimg = cv::imread("../../data/images/color/ctest0.png", cv::IMREAD_COLOR);

    EXPECT_CALL(resMgr, loadDepthImage("test0.png", cv::IMREAD_UNCHANGED)).WillOnce(Return(dimg));
    EXPECT_CALL(resMgr, loadColorImage("ctest0.png", cv::IMREAD_COLOR)).WillOnce(Return(cimg));

    SimpleImageProducer frame_producer(&resMgr, 0);
    frame_producer.initialize();
    FrameElement* frame_element = frame_producer.generateCurrentFrame();

}


TEST (longcudaContourOperationUnitTest, ContourOperation) {
    /// Initialize Application context
    AppContextBuilder app_ctx_builder;
    app_ctx_builder.setViewPortDimensions(800, 640);
    app_ctx_builder.setWindowTitle("Edge App");
    app_ctx_builder.setResDir("../../data");

    AppContext* const app_ctx = app_ctx_builder.Build();

    DepthImagePolicy dimg_policy(app_ctx);

    ResMgr* resMgr = app_ctx->getResMgr();
    SimpleImageProducer frame_producer(resMgr, 0);
    frame_producer.initialize();
    FrameElement* frame_element = frame_producer.generateCurrentFrame();

    DepthImagePolicy depth_policy(app_ctx);
    depth_policy.intialize();
    depth_policy.setFrameData(frame_element);
    depth_policy.executePolicy();

    ContourAttributes* contour_attribtues = depth_policy.getContourAttributes();

    LineSegmentContourPolicy contour_policy;
    contour_policy.setContourData(contour_attribtues);
    contour_policy.executePolicy();


}
