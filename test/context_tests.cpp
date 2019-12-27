//
// Created by ynki9 on 12/27/19.
//
#include "gtest/gtest.h"
#include "context.h"


TEST (AppContextBuilderUnitTest, AssembleContext) {
    AppContext::Builder ctx_builder;
    ctx_builder.initializeCuda();

    EXPECT_TRUE(ctx_builder.isCudaInitialized());
}
