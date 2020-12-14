//
// Created by ynki9 on 12/8/20.
//

#ifndef PROJECT_EDGE_VISUALIZATIONWINDOW_H
#define PROJECT_EDGE_VISUALIZATIONWINDOW_H

#include "gui/imgui_all.h"

struct WindowPlacementInfo {
    ImVec2 Size;
    ImVec2 Position;
};

class VisualizationWindow {
public:
    virtual void Show(WindowPlacementInfo placementInfo) = 0;
    virtual const char *GetTitle() const = 0;

    VisualizationWindow() = default;
    virtual ~VisualizationWindow() = default;
    VisualizationWindow(const VisualizationWindow &) = delete;
    VisualizationWindow(const VisualizationWindow &&) = delete;
    VisualizationWindow &operator=(const VisualizationWindow &) = delete;
    VisualizationWindow &operator=(const VisualizationWindow &&) = delete;

};


#endif //PROJECT_EDGE_VISUALIZATIONWINDOW_H
