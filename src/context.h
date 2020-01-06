//
// Created by ynki9 on 12/26/19.
//

#ifndef PROJECT_EDGE_CONTEXT_H
#define PROJECT_EDGE_CONTEXT_H

using CudaDevice = int;

class AppContext {
public:
    class Builder;

    AppContext(const CudaDevice &cuda_device):
            cuda_device_(cuda_device) {}

    inline CudaDevice getCudaDevice(){
        return this->cuda_device_;
    }

private:
    const CudaDevice cuda_device_;
};


/*****************************************************/
class AppContext::Builder {
public:
    /// Default Constructor
    Builder():
    cuda_device_(-1){}

    /**
     * Builds an application context
     * @return
     */
    AppContext* const Build();
    // void initializeOpenGL();

    void initializeCuda();
    bool isCudaInitialized(){return this->cuda_device_ != -1;}

private:
    CudaDevice cuda_device_;
};



#endif //PROJECT_EDGE_CONTEXT_H
