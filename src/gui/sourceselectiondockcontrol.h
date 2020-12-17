//
// Created by ynki9 on 12/15/20.
//

#ifndef PROJECT_EDGE_SOURCESELECTIONDOCKCONTROL_H
#define PROJECT_EDGE_SOURCESELECTIONDOCKCONTROL_H

#include "dataflow/pipeline_filter.h"
#include "frame.h"
#include "context/context.h"

// System Deps
#include <memory>

#include "gui/dockcontrol.h"
#include "filter/filtermanager.h"
#include "gui/configcontrollerinterface.h"


class SourceSelectionDockControl : public DockControl {
public:
    SourceSelectionDockControl(AppContext &ctxt):
    app_context_(ctxt){
        source_types_.push_back(pair<int, string>(static_cast<int>(PipelineFilterTypes::FILESOURCEPROVIDER), "Local Image"));
        source_types_.push_back(pair<int, string>(static_cast<int>(PipelineFilterTypes::K4ACAMERAPROVIDER), "K4A Camera"));
    };
    ~SourceSelectionDockControl() override = default;

    DockControlStatus Show();

    SourceSelectionDockControl(const SourceSelectionDockControl &) = delete;
    SourceSelectionDockControl(const SourceSelectionDockControl &&) = delete;
    SourceSelectionDockControl operator=(const SourceSelectionDockControl &) = delete;
    SourceSelectionDockControl operator=(const SourceSelectionDockControl &&) = delete;

private:
    void ShowSourceDockControls();
    void RefreshDevice();

    std::vector<std::pair<int, std::string>> source_types_;
    std::shared_ptr<ConfigControllerInterface> current_source_controller_;
    int selected_source_id_;
    AppContext &app_context_;
};


#endif //PROJECT_EDGE_SOURCESELECTIONDOCKCONTROL_H
