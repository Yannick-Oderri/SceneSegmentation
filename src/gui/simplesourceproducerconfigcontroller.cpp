//
// Created by ynki9 on 12/15/20.
//
#include "gui/imgui_all.h"

#include "simplesourceproducerconfigcontroller.h"



void SimpleSourceProducerConfigController::Show() {
    bool modified = false;

    ImGui::NewLine();
    ImGui::Separator();
    ImGui::NewLine();


    if (ImGui::TreeNode("File-Source Params"))
    {
        modified = modified | ImGui::InputInt("Image Index", &m_SourceConfig.imageID_, 1, 1);
        modified = modified | ImGui::InputInt("Sample Rate", &m_SourceConfig.updateRate_, 1, 1);

        ImGui::TreePop();
    }

    if (modified){
        for (auto observer : m_Obervers){
            observer->NotifyData(m_SourceConfig);
        }
    }

}

void SimpleSourceProducerConfigController::RegisterObserver(Observer<SimpleImageProducerConfig> &observer) {
    m_Obervers.push_back(&observer);
}