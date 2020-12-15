//
// Created by ynki9 on 12/7/20.
//

#ifndef PROJECT_EDGE_WINDOWDOCK_H
#define PROJECT_EDGE_WINDOWDOCK_H

#include <memory>
#include <stack>
#include <string>

#include "gui/imgui_all.h"
#include "gui/dockcontrol.h"

class WindowDock
{
public:
    enum class Edge
    {
        Left,
        Right,
        Top,
        Bottom
    };

    WindowDock(Edge edge);
    void PushDockControl(std::unique_ptr<DockControl> &&dockControl);
    void Show(ImVec2 regionPosition, ImVec2 regionSize);
    ImVec2 GetSize();

private:
    void SetRegion(ImVec2 position, ImVec2 size);
    void SetSize(ImVec2 size);

    static constexpr ImGuiWindowFlags DockWindowFlags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                                        ImGuiWindowFlags_AlwaysAutoResize |
                                                        ImGuiWindowFlags_NoTitleBar |
                                                        ImGuiWindowFlags_NoBringToFrontOnFocus |
                                                        ImGuiWindowFlags_HorizontalScrollbar;

    std::stack<std::unique_ptr<DockControl>> m_dockControls;

    Edge m_edge;
    std::string m_windowName;

    // The region into which the dock is allowed to draw
    //
    ImVec2 m_regionPosition = ImVec2(0.f, 0.f);
    ImVec2 m_regionSize = ImVec2(0.f, 0.f);

    // The actual size/location of the dock window, in absolute window coordinates.
    // Must be within by m_region*
    //
    ImVec2 m_size = ImVec2(0.f, 0.f);

    bool m_isResizing = false;
    bool m_userHasResized = false;
};

#endif //PROJECT_EDGE_WINDOWDOCK_H
