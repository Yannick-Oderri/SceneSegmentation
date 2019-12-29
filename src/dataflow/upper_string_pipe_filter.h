//
// Created by Yannick Roberts on 2019-12-29.
//

#ifndef PROJECT_EDGE_UPPER_STRING_PIPE_FILTER_H
#define PROJECT_EDGE_UPPER_STRING_PIPE_FILTER_H

#define UPPER_STRING_FILTER "upper_string"

class UpperStringPipeFilter: protected PipeFilter<std::string> {
public:
    UpperStringPipeFilter(QueueClient<std::string>* const in_queue, QueueClient<std::string>* const out_queue):
            PipeFilter<cv::Mat>(in_queue, out_queue){};

    void start();
};


#endif //PROJECT_EDGE_UPPER_STRING_PIPE_FILTER_H
