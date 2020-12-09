//
// Created by ynki9 on 12/7/20.
//
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
        AppContext* const app_ctx = app_ctx_builder.Build();
    }

    // void TearDown() override{}

    AppContext* app_ctx;
};


TEST(ViewerAppTest, create) {
    ViewerApp viewer(app_ctx);
}