//
// Created by ynki9 on 12/16/20.
//

#include <memory>

#include "gtest/gtest.h"
#include "context/context_factory.h"
#include "filter/filtermanager.h"

class FilterManagerTest: public ::testing::Test {
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

TEST_F(FilterManagerTest, createInstance) {
    FilterHandle handle = -1;
    FilterManager::FilterCreationResults res =  FilterManager::Instance().AppendFilter(
            *app_ctx.get(),
            PipelineFilterTypes::SOURCEPROVIDEREND,
            handle);
    ASSERT_EQ(res, FilterManager::FAILED);


    res =  FilterManager::Instance().AppendFilter(
            *app_ctx.get(),
            PipelineFilterTypes::K4ACAMERAPROVIDER,
            handle);
    ASSERT_EQ(res, FilterManager::FAILED);


    res =  FilterManager::Instance().AppendFilter(
            *app_ctx.get(),
            PipelineFilterTypes::FILESOURCEPROVIDER,
            handle);
    ASSERT_EQ(res, FilterManager::SUCCESS);
}

