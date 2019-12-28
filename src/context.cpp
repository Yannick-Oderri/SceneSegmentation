//
// Created by ynki9 on 12/26/19.
//

#include "context.h"
#include <boost/log/trivial.hpp>
#include <cuda_runtime.h>
#include <helper_cuda.h>


AppContext* AppContext::Builder::Build() const {
    return nullptr;
}

void AppContext::Builder::initializeCuda() {
    BOOST_LOG_TRIVIAL(info) << "Initializing Nvidia CUDA";

    CudaDevice devID = -1;


    devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(devID));
    int major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
    BOOST_LOG_TRIVIAL(info) << "GPU Device " << devID << _ConvertSMVer2ArchName(major, minor)
                            << "with compute capability" << major << "." << minor;

    this->cuda_device_ = devID;
}
