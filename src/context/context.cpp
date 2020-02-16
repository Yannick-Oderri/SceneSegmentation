//
// Created by ynki9 on 12/26/19.
//

#include "context.h"
#include <boost/log/trivial.hpp>
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

AppContext::AppContext(AppContext::Builder &ctx_builder):
        cuda_device_(ctx_builder.cuda_device_),
        window_width_(ctx_builder.window_width_),
        window_height_(ctx_builder.window_height_),
        window_(ctx_builder.window_),
        res_mgr_(ctx_builder.res_mgr_){}


AppContext* const AppContext::Builder::Build() {
    if (this->initializeCuda() != 0){
        BOOST_LOG_TRIVIAL(error) << "Unable to initialize CUDA.";
        return nullptr;
    }

    if (this->initializeMainWindow() != 0){
        BOOST_LOG_TRIVIAL(error) << "Unable to initalize Opengl Window.";
        return nullptr;
    }

    ResMgr* res_mgr = this->initializeResMgr();


    AppContext* const appContext = new AppContext(*this, res_mgr);

    return appContext;
}

int AppContext::Builder::initializeCuda() {
    BOOST_LOG_TRIVIAL(info) << "Initializing Nvidia CUDA";

#ifdef WITH_CUDA
    CudaDevice devID = -1;


    devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(devID));
    int major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
    BOOST_LOG_TRIVIAL(info) << "GPU Device " << devID << _ConvertSMVer2ArchName(major, minor)
                            << "with compute capability" << major << "." << minor;
    if(major == 10){
        BOOST_LOG_TRIVIAL(error) << "Invalid CUDA version";
        return -1;
    }

    this->cuda_device_ = devID;
    return 0;
#else
    BOOST_LOG_TRIVIAL(info) << "WITH_CUDA compile flag not set";
#endif
}

int AppContext::Builder::initializeMainWindow() {
    if (this->window_height_ < 0 || this->window_width_ < 0) {
        BOOST_LOG_TRIVIAL(error) << "Viewport dimensions not initialized" ;
        return -1;
    }

    glfwInit();
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );
    glfwWindowHint( GLFW_RESIZABLE, GL_FALSE );
    window_ = glfwCreateWindow( window_width_, window_height_, this->window_title_.data(), nullptr, nullptr );
    if ( nullptr == window_ )
    {
        BOOST_LOG_TRIVIAL(error) << "Failed to create GLFW window";
        glfwTerminate( );
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent( window_ );

    glewExperimental = GL_TRUE;
    if ( GLEW_OK != glewInit( ) )
    {
        BOOST_LOG_TRIVIAL(error) << "Failed to initialize GLEW" ;
        return EXIT_FAILURE;
    }
    int screen_width, screen_height;
    glfwGetFramebufferSize( window_, &screen_width, &screen_height );
    glViewport( 0, 0, screen_width, screen_height );
    glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    return 0;
}


int AppContext::Builder::initializeResMgr() {
    // validate resource directory
    if (res_dir_.empty()){
        BOOST_LOG_TRIVIAL(error) << "Invalid Resource directory.";
        return -1;
    }

    this->res_mgr_ = new ResMgr(*this);
}