#include <thread>
#include <GL/glew.h>    // Initialize with glewInit()
#include <GLFW/glfw3.h>
#include <imgui.h>
#include "context/context_factory.h"
#include "filter/simple_image_producer.h"
#include "viewer/gl_viewport.h"
#include "res/resource_mgr.h"
#include "filter/gl_edge_disc_filter.h"
#include "filter/gl_depth_img_filter.h"
#include "filter/contour_processor.h"
#include "component/contour_policy.h"

#ifndef AS_LIB

void generateProcessingPipeline(AppContext* const context){
    /// Image pipeline loader
    SimpleImageProducer producer(context->getResMgr(), 0, 33);
    producer.initialize();

    /// Depth Image Pipeline
    GLDepthImageFilter dpt_img_fltr(producer.getOutQueue(), new DepthImagePolicy(context));
    dpt_img_fltr.initialize();

    /// Contour Filter Pipeline
    ContourProcessorPipeFilter contour_filter(dpt_img_fltr.getOutQueue(), new LineSegmentContourPolicy());
    contour_filter.initialize();


    std::thread producer_thread(&SimpleImageProducer::start, &producer);
    std::thread dpt_fltr_thread(&GLDepthImageFilter::start, &dpt_img_fltr);
    std::thread contour_fltr_thread(&ContourProcessorPipeFilter::start, &contour_filter);

    /// End Pipeline
    // producer.signalEnd();
    producer_thread.join();
}

int main(int argc, char **argv){

    /// Initialize Application context
    AppContextBuilder app_ctx_builder;
    app_ctx_builder.setViewPortDimensions(800, 640);
    app_ctx_builder.setWindowTitle("Edge App");
    app_ctx_builder.setResDir("../data");

    AppContext* const app_ctx = app_ctx_builder.Build();


    generateProcessingPipeline(app_ctx);

    return 0;


    /// Pointcloud Renderer
    GLViewport viewport(*app_ctx);


    /// Image pipeline loader
    SimpleImageProducer producer(app_ctx->getResMgr(), 0);
    producer.initialize();
    std::thread producer_tread(&SimpleImageProducer::start, &producer);

    bool end_loop = false;
    int dims[3] = {480, 640,3};
    vector<cv::Point3f> points(480*640);
    cv::Mat xyz(480, 640, CV_32FC3);

    cv::Mat rgb = cv::imread("../data/depth/ctest55.png", -1);

    while(end_loop == false){
        producer.getOutQueue()->waitData();

        FrameElement* const frame_element = producer.getOutQueue()->front();
        cv::Mat depth_data = frame_element->getDepthFrameData()->getcvMat();

        for (int r = 0; r < depth_data.rows; r++){
            for (int c = 0; c < depth_data.cols; c++){
                float x, y, z;
                frame_element->getDepthFrameData()->getXYZPoint(r, c, x, y, z);
                points[r*depth_data.rows+c] = cv::Point3f(x, y, z);
                // xyz.at<cv::Point3d>(r, c) = point;
            }
        }
        producer.getOutQueue()->pop();
        xyz = cv::Mat(points).reshape(3, 480);
        viewport.RenderFrame(rgb, xyz);
    }

    producer.signalEnd();
    producer_tread.join();

}

#endif