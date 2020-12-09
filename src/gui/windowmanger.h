//
// Created by ynki9 on 12/7/20.
//

#ifndef PROJECT_EDGE_WINDOWMANGER_H
#define PROJECT_EDGE_WINDOWMANGER_H

#include "gui/dockcontrol.h"
#include "gui/visualizationwindow.h"
#include "gui/windowdock.h"

class WindowManger {
public:
    static WindowManager &Instance();

    void setGLWindowSize(ImVec2 glWindowSize);
    void SetMenuBarHeight(float menuBarHeight);

    void AddWindow(std::unique_ptr<IK4AVisualizationWfaindow> &&window);
    void AddWindowGroup(std::vector<std::unique_ptr<IK4AVisualizationWindow>> &&windowGroup);
    void ClearFullscreenWindow();
    void ClearWindows();

    void PushLeftDockControl(std::unique_ptr<IK4ADockControl> &&dockControl);
    void PushBottomDockControl(std::unique_ptr<IK4ADockControl> &&dockControl);

    void ShowAll();

    WindowManager(const K4AWindowManager &) = delete;
    WindowManager &operator=(const K4AWindowManager &) = delete;
    WindowManager(const K4AWindowManager &&) = delete;
    WindowManager &operator=(const K4AWindowManager &&) = delete;

private:
    WindowManager() = default;
    struct WindowListEntry
    {
        WindowListEntry() : IsWindowGroup(true) {}

        WindowListEntry(std::unique_ptr<IK4AVisualizationWindow> &&window) :
                IsWindowGroup(false),
                Window(std::move(window))
        {
        }

        WindowListEntry(std::vector<std::unique_ptr<VisualizationWindow>> &&windowGroup) : IsWindowGroup(true)
        {
            for (auto &&windowEntry : windowGroup)
            {
                WindowGroup.emplace_back(std::move(windowEntry));
            }
        }

        bool IsWindowGroup;
        std::unique_ptr<VisualizationWindow> Window;
        std::vector<WindowListEntry> WindowGroup;
    };

};


#endif //PROJECT_EDGE_WINDOWMANGER_H
