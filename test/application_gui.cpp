//
// Created by ynki9 on 12/7/20.
//
#include <memory>

#include "gtest/gtest.h"
#include "context/context_factory.h"
#include "gui/application.hpp"

class ViewerAppTest: public ::testing::Test {
protected:
    void SetUp() override {
        /// Initialize Application context
        AppContextBuilder app_ctx_builder;
        app_ctx_builder.setViewPortDimensions(800, 640);
        app_ctx_builder.setWindowTitle("Edge App");
        app_ctx_builder.setResDir("../../data");
        app_ctx_builder.setOutDir("./results");
        app_ctx = app_ctx_builder.Build();
    }

    // void TearDown() override{}

    std::unique_ptr<AppContext> app_ctx;
};


TEST_F(ViewerAppTest, create) {
    ViewerApp viewer(app_ctx);

    viewer.Initialize();
    viewer.Run();
}