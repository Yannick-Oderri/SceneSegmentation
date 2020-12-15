//
// Created by ynki9 on 12/8/20.
//

#ifndef PROJECT_EDGE_DOCKCONTROL_H
#define PROJECT_EDGE_DOCKCONTROL_H

enum class DockControlStatus
{
    Ok,
    ShouldClose
};

class DockControl {
public:
    virtual DockControlStatus Show() = 0;

    DockControl() = default;
    virtual ~DockControl() = default;

    DockControl(const DockControl &) = delete;
    DockControl &operator=(const DockControl &) = delete;
    DockControl(const DockControl &&) = delete;
    DockControl &operator=(const DockControl &&) = delete;
};


#endif //PROJECT_EDGE_DOCKCONTROL_H
