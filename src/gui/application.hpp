//
// Created by ynki9 on 12/7/20.
//

#ifndef PROJECT_EDGE_GLVIEWER_H
#define PROJECT_EDGE_GLVIEWER_H

#include "context/context.h"
#include <memory.h>


class ViewerApp {
public:
    explicit ViewerApp(std::unique_ptr<AppContext> ctxt): appContext_(std::move(ctxt)){};
    ~glViewer();

    void Run();

    ViewerApp(const ViewerApp &) = delete;
    ViewerApp(const ViewerApp &&) = delete;
    ViewerApp &operator=(const ViewerApp &) = delete;
    ViewerApp &operator=(const ViewerApp &&) = delete;

private:
    void ShowMainMenuBar();

    void setHighDpi();
    void ShowErrorOverlay();

    bool showPerfMetrics = false;
    std::shared_ptr<AppContext> appContext_;
};


#endif //PROJECT_EDGE_GLVIEWER_H
