//
// Created by ynki9 on 12/7/20.
//

#ifndef PROJECT_EDGE_GLVIEWER_H
#define PROJECT_EDGE_GLVIEWER_H

// system headers
#include <memory>

// project headers
#include "context/context.h"
#include "gui/viewersettingsmanager.h"



class ViewerApp {
public:
    ViewerApp(std::unique_ptr<AppContext> &ctxt): appContext_(std::move(ctxt)){};
    ~ViewerApp(){};

    void Run();

    ViewerApp(const ViewerApp &) = delete;
    ViewerApp(const ViewerApp &&) = delete;
    ViewerApp &operator=(const ViewerApp &) = delete;
    ViewerApp &operator=(const ViewerApp &&) = delete;

private:
    void ShowMainMenuBar();

    void setHighDpi();
    void ShowErrorOverlay();    
    void ShowViewerOptionMenuItem(const char *msg, ViewerOption option);


    bool showPerfMetrics = false;
    std::unique_ptr<AppContext> appContext_;
};




#endif //PROJECT_EDGE_GLVIEWER_H
