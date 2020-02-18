#include <thread>
#include <GL/glew.h>    // Initialize with glewInit()
#include <GLFW/glfw3.h>
#include <imgui.h>
#include "context/context_factory.h"
#include "simple_image_producer.h"
#include "viewer/gl_viewport.h"
#include "res/resource_mgr.h"

#ifndef AS_LIB

int main(int argc, char **argv){

    /// Initialize Application context
    AppContextBuilder app_ctx_builder;
    app_ctx_builder.setViewPortDimensions(800, 640);
    app_ctx_builder.setWindowTitle("Edge App");
    app_ctx_builder.setResDir("../data");

    AppContext* const app_ctx = app_ctx_builder.Build();


    /// Pointcloud Renderer
    GLViewport viewport(*app_ctx);


    /// Image pipeline loader
    SimpleImageProducer producer;
    producer.initialize();
    std::thread producer_tread(&SimpleImageProducer::start, &producer);

    bool end_loop = false;
    int dims[3] = {480, 640,3};
    vector<cv::Point3d> points(480*640);
    cv::Mat xyz(480, 640, CV_32FC3);

    cv::Mat rgb = cv::imread("../data/depth/ctest0.png", -1);

    while(end_loop == false){
        producer.getOutQueue()->waitData();

        FrameElement* const frame_element = producer.getOutQueue()->front();
        cv::Mat depth_data = frame_element->getDepthFrameData()->getcvMat();

        for (int r = 0; r < depth_data.rows; r++){
            for (int c = 0; c < depth_data.cols; c++){
                double x, y, z;
                frame_element->getDepthFrameData()->getXYZPoint(r, c, x, y, z);
                points[r*depth_data.rows+c] = cv::Point3d(x, y, z);
                //cv::Point3d point(x,y,z);
                //xyz.at<cv::Point3d>(r, c) = point;
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