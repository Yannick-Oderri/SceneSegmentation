//
// Created by ynki9 on 12/26/19.
//

#include "context.h"
#include <boost/log/trivial.hpp>

AppContext* AppContext::Builder::Build() const {
    return nullptr;
}

void AppContext::Builder::initializeCuda() {
    BOOST_LOG_TRIVIAL(info) << "Initializing Nvidia CUDA";
    this->cuda_device_ = -1;
}
