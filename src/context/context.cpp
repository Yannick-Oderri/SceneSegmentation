//
// Created by ynki9 on 12/26/19.
//

#include "context/context_factory.h"

AppContext::AppContext(AppContextBuilder &ctx_builder):
        cuda_device_(ctx_builder.cuda_device_),
        window_width_(ctx_builder.window_width_),
        window_height_(ctx_builder.window_height_),
        window_(ctx_builder.window_),
        res_mgr_(ctx_builder.res_mgr_){}


