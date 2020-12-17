//
// Created by ynki9 on 12/16/20.
//

#ifndef PROJECT_EDGE_CONFIGCONTROLLERINTERFACE_H
#define PROJECT_EDGE_CONFIGCONTROLLERINTERFACE_H


class ConfigControllerInterface {
public:
    virtual void Show() = 0;
    virtual void UnregisterObservers() = 0;
    virtual ~ConfigControllerInterface() = default;
};


#endif //PROJECT_EDGE_CONFIGCONTROLLERINTERFACE_H
