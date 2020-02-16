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
    app_ctx_builder.setResDir("../../data");

    AppContext* const app_ctx = app_ctx_builder.Build();


    /// Pointcloud Renderer
    GLViewport viewport(*app_ctx);



    SimpleImageProducer producer;
    producer.initialize();
    std::thread producer_tread(&SimpleImageProducer::start, &producer);
}

#endif