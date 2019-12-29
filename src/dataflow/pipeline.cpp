//
// Created by ynki9 on 12/26/19.
//

#include "pipeline.h"
#include "pipeline_filter.h"
// #include "cuda_sobel_pipe_filter.h"

void Pipeline::AppendFilter(PipeFilterType filter_type) {
    AbstractPipeFilter* pipe_filter;
    switch (filter_type){
    }
}