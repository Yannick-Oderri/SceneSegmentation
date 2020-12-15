//
// Created by ynki9 on 12/14/20.
//

#include "gui/viewersettingsmanager.h"

ViewerOptions::ViewerOptions() {
    static_assert(static_cast<size_t>(ViewerOption::MAX) == 4, "Need to add a new viewer option default");

    Options[static_cast<size_t>(ViewerOption::ShowFrameRateInfo)] = false;
    Options[static_cast<size_t>(ViewerOption::ShowInfoPane)] = true;
    Options[static_cast<size_t>(ViewerOption::ShowLogDock)] = false;
    Options[static_cast<size_t>(ViewerOption::ShowDeveloperOptions)] = false;
}