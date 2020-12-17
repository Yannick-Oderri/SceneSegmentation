//
// Created by ynki9 on 12/15/20.
//

#ifndef PROJECT_EDGE_SIMPLE_SOURCE_PRODUCER_CONFIG_H
#define PROJECT_EDGE_SIMPLE_SOURCE_PRODUCER_CONFIG_H

class SimpleSourceProducerConfig{
public:
    bool isDirty();

private:
    void reset();

    bool m_isDirty;
    int m_imageID;
    int m_updateRate;
};

#endif //PROJECT_EDGE_SIMPLE_SOURCE_PRODUCER_CONFIG_H
