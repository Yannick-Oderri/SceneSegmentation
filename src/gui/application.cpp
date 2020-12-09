//
// Created by ynki9 on 12/7/20.
//

#include "application.hpp"


void ViewerApp::Run() {
    while (!glfwWindowShouldClose(appContext_->getGLContext())){
        glfwPollEvents();

        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ShowMainMenuBar();

    }
}


void ViewerApp::ShowMainMenuBar()
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("Settings"))
        {
            ShowViewerOptionMenuItem("Show log dock", ViewerOption::ShowLogDock);
            ShowViewerOptionMenuItem("Show info overlay", ViewerOption::ShowInfoPane);

            if (K4AViewerSettingsManager::Instance().GetViewerOption(ViewerOption::ShowInfoPane))
            {
                ShowViewerOptionMenuItem("Show framerate", ViewerOption::ShowFrameRateInfo);
            }

            ShowViewerOptionMenuItem("Show developer options", ViewerOption::ShowDeveloperOptions);

            ImGui::Separator();

            if (ImGui::MenuItem("Load default settings"))
            {
                //K4AViewerSettingsManager::Instance().SetDefaults();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Quit"))
            {
                glfwSetWindowShouldClose(m_window, true);
            }
            ImGui::EndMenu();
        }

//        if (K4AViewerSettingsManager::Instance().GetViewerOption(ViewerOption::ShowDeveloperOptions))
//        {
//            if (ImGui::BeginMenu("Developer"))
//            {
//                ImGui::MenuItem("Show demo window", nullptr, &m_showDemoWindow);
//                ImGui::MenuItem("Show style editor", nullptr, &m_showStyleEditor);
//                ImGui::MenuItem("Show metrics window", nullptr, &m_showMetricsWindow);
//                ImGui::MenuItem("Show perf counters", nullptr, &m_showPerfCounters);
//
//                ImGui::EndMenu();
//            }
//        }

//        K4AWindowManager::Instance().SetMenuBarHeight(ImGui::GetWindowSize().y);
        ImGui::EndMainMenuBar();
    }
}