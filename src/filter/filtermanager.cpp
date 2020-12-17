//
// Created by ynki9 on 12/15/20.
//

#include "context/context.h"

#include "filtermanager.h"

#include "filter/simple_image_producer.h"
#include "gui/simplesourceproducerconfigcontroller.h"

FilterManager::FilterCreationResults FilterManager::AppendFilter(AppContext &appContext, PipelineFilterTypes filterType, FilterHandle &handle) {
    bool isValid = FilterManager::ValidateFilterAppend(filterType);
    FilterCreationResults results =  FAILED;
    switch (filterType) {
        case PipelineFilterTypes::FILESOURCEPROVIDER :
            results = FilterManager::CreateSimpleFileSourceFilter(appContext);
            break;
        case PipelineFilterTypes::K4ACAMERAPROVIDER :
            //results = FilterManager::CreateK4ASourceFilter();
            break;
    }

    if (results == SUCCESS) {
        handle = m_pipeline.size();
    }

    return results;
}

bool FilterManager::ValidateFilterAppend(PipelineFilterTypes filterType) {
    if (m_pipeline.size() == 0 &&
        static_cast<int>(filterType) >= static_cast<int>(PipelineFilterTypes::SOURCEPROVIDEREND)) {
        return false;
    }

    return true;
}


FilterManager::FilterCreationResults FilterManager::CreateSimpleFileSourceFilter(AppContext &appContext) {
    SimpleImageProducerConfig config = {true,0, 33};

    // Initialize source filter
    std::shared_ptr<SimpleImageProducer> sourceFilter = std::make_shared<SimpleImageProducer>(appContext.getResMgr(), config);
    // Initialize config controller
    std::shared_ptr<SimpleSourceProducerConfigController> controller = std::make_shared<SimpleSourceProducerConfigController>(config);
    // Register Filter to Controller for configuration changes
    controller->RegisterObserver(*sourceFilter.get());

    // Store Controller and Filter
    m_pipeline.push(sourceFilter);
    m_filterControllers.emplace_back(controller);
}

std::shared_ptr<ConfigControllerInterface> FilterManager::RequestConfigController(FilterHandle handle) {
    if (handle >= m_pipeline.size()) {
        return nullptr;
    }

    return m_filterControllers[handle];
}



void FilterManager::UnloadFilters() {
    for (auto controller : m_filterControllers) {
        controller->UnregisterObservers();
    }

    while (!m_pipeline.empty()) {
        m_pipeline.pop();
    }
    m_filterControllers.clear();
}