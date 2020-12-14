//
// Created by ynki9 on 12/7/20.
//

#include  "gui/imgui_all.h"
#include "gui/application.hpp"
#include "gui/windowmanager.h"



const ImVec4 ClearColor(0.01f, 0.01f, 0.01f, 1.0f);

constexpr int GlfwFailureExitCode = -1;

constexpr float HighDpiScaleFactor = 2.0f;

void ViewerApp::Run() {
    while (!glfwWindowShouldClose(appContext_->getGLContext())){
        glfwPollEvents();

        // Start the ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ShowMainMenuBar();

        WindowManager::Instance().ShowAll();


        // Finalize/render frame
        //
        ImGui::Render();
        int displayW;
        int displayH;
        glfwMakeContextCurrent(appContext_->getGLContext());
        glfwGetFramebufferSize(appContext_->getGLContext(), &displayW, &displayH);
        glViewport(0, 0, displayW, displayH);
        WindowManager::Instance().SetGLWindowSize(ImVec2(float(displayW), float(displayH)));
        glClearColor(ClearColor.x, ClearColor.y, ClearColor.z, ClearColor.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(appContext_->getGLContext());
    }
}


void ViewerApp::ShowMainMenuBar()
{
    if (ImGui::BeginMainMenuBar())
    {
        ImGui::EndMainMenuBar();
    }
}

void ViewerApp::ShowViewerOptionMenuItem(const char *msg, ViewerOption option)
{
    auto &settings = ViewerSettingsManager::Instance();
    bool isSet = settings.GetViewerOption(option);

    if (ImGui::MenuItem(msg, nullptr, isSet))
    {
        //settings.SetViewerOption(option, !isSet);
    }
}