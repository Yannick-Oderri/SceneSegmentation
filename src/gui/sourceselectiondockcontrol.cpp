//
// Created by ynki9 on 12/15/20.
//

#include <boost/log/trivial.hpp>

#include "sourceselectiondockcontrol.h"
#include "gui/imgui_all.h"
#include "gui/imguiextensions.h"

DockControlStatus SourceSelectionDockControl::Show() {
    ImGui::SetNextTreeNodeOpen(true, ImGuiCond_FirstUseEver);
    if (ImGui::TreeNode("Select Source Device")) {
        if (ImGuiExtensions::ComboBox("Data Source",
                "(No Sources Available)",
                ImGuiComboFlags_None,
                source_types_,
                &selected_source_id_)) {

            FilterManager::Instance().UnloadFilters();
            FilterHandle fhandle = -1;

            if (FilterManager::Instance().AppendFilter(app_context_, static_cast<PipelineFilterTypes>(selected_source_id_), fhandle) == !FilterManager::SUCCESS) {
                BOOST_LOG_TRIVIAL(error) << "Failed to load Filter Type: " << selected_source_id_;
                return DockControlStatus::Ok;
            }

            current_source_controller_ = FilterManager::Instance().RequestConfigController(fhandle);
        }

        ShowSourceDockControls();

        ImGui::TreePop();
    }

    return DockControlStatus::Ok;
}

void SourceSelectionDockControl::ShowSourceDockControls(){
    if (this->current_source_controller_ != nullptr) {
        current_source_controller_->Show();
    }
}

void SourceSelectionDockControl::RefreshDevice() {
    //ASSERT(false);
}