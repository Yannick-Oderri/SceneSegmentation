//
// Created by ynki9 on 12/15/20.
//

#ifndef PROJECT_EDGE_SIMPLESOURCEPRODUCERCONFIGCONTROLLER_H
#define PROJECT_EDGE_SIMPLESOURCEPRODUCERCONFIGCONTROLLER_H

#include <vector>
#include <memory>

#include <filter/simple_image_producer.h>
#include "dataflow/observer.h"
#include "gui/configcontrollerinterface.h"



class SimpleSourceProducerConfigController : public ConfigControllerInterface {
public:
    SimpleSourceProducerConfigController(SimpleImageProducerConfig config):
            m_SourceConfig(config),
            m_Obervers(){}

    void Show();

    void RegisterObserver(Observer<SimpleImageProducerConfig> &observer);

private:

    SimpleImageProducerConfig m_SourceConfig;
    std::vector<Observer<SimpleImageProducerConfig>*> m_Obervers;
};


#endif //PROJECT_EDGE_SIMPLESOURCEPRODUCERCONFIGCONTROLLER_H
