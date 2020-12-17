//
// Created by ynki9 on 12/15/20.
//

#ifndef PROJECT_EDGE_FILTERFACTORY_H
#define PROJECT_EDGE_FILTERFACTORY_H

#include "dataflow/pipeline_filter.h"
#include "gui/configcontrollerinterface.h"

#include <queue>
#include <vector>
#include <memory>

typedef int FilterHandle;

enum class PipelineFilterTypes{
    FILESOURCEPROVIDER,
    K4ACAMERAPROVIDER,

    SOURCEPROVIDEREND,

    MAX
};


class FilterManager {
public:
    static FilterManager &Instance(){
        static FilterManager instance;
        return instance;
    }


    enum FilterCreationResults{
        SUCCESS,
        FAILED,

    };

    FilterCreationResults AppendFilter(AppContext &appContext, PipelineFilterTypes filterType, FilterHandle &handle);
    std::shared_ptr<ConfigControllerInterface> RequestConfigController(FilterHandle filter);
    void UnloadFilters();

private:
    // Constructor
    FilterManager():
    m_pipeline(std::queue<std::shared_ptr<AbstractPipeFilter>>())
    {};


    FilterCreationResults CreateSimpleFileSourceFilter(AppContext &appContext);
    bool ValidateFilterAppend(PipelineFilterTypes filterType);


    std::queue<std::shared_ptr<AbstractPipeFilter>> m_pipeline;
    std::vector<std::shared_ptr<ConfigControllerInterface>> m_filterControllers;
};


#endif //PROJECT_EDGE_FILTERFACTORY_H
