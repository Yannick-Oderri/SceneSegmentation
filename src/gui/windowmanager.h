//
// Created by ynki9 on 12/7/20.
//

#ifndef PROJECT_EDGE_WINDOWMANGER_H
#define PROJECT_EDGE_WINDOWMANGER_H

#include <vector>
#include <memory>

#include "gui/dockcontrol.h"
#include "gui/visualizationwindow.h"
#include "gui/windowdock.h"

class WindowManager {
public:
    static WindowManager &Instance();

    void SetGLWindowSize(ImVec2 glWindowSize);
    void SetMenuBarHeight(float menuBarHeight);

    void AddWindow(std::unique_ptr<VisualizationWindow> &&window);
    void AddWindowGroup(std::vector<std::unique_ptr<VisualizationWindow>> &&windowGroup);
    void ClearFullscreenWindow();
    void ClearWindows();

    void PushLeftDockControl(std::unique_ptr<DockControl> &&dockControl);
    void PushBottomDockControl(std::unique_ptr<DockControl> &&dockControl);

    void ShowAll();

    WindowManager(const WindowManager &) = delete;
    WindowManager &operator=(const WindowManager &) = delete;
    WindowManager(const WindowManager &&) = delete;
    WindowManager &operator=(const WindowManager &&) = delete;

private:
    WindowManager() = default;
    struct WindowListEntry
    {
        WindowListEntry() : IsWindowGroup(true) {}

        WindowListEntry(std::unique_ptr<VisualizationWindow> &&window) :
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

    ImVec2 glWindowSize_ = {0, 0};
    float menuBarHeight_ = 0;

    VisualizationWindow *maximizedWindow_ = nullptr;
    WindowListEntry windows_;   

    WindowDock leftDock_ = WindowDock(WindowDock::Edge::Left);
    WindowDock bottomDock_ = WindowDock(WindowDock::Edge::Bottom);
};


#endif //PROJECT_EDGE_WINDOWMANGER_H
