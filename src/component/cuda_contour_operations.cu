
#ifndef CUDA_CONTOUR_OPERATOINS_H
#define CUDA_CONTOUR_OPERATOINS_H

//#include "utils/cuda_defs.cu"
#include "cu_contour_opr.h"
#include <helper_functions.h>
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include <math.h>

using namespace std;

class Mat2{
    float vals[4];

public:
    __device__
    Mat2(float c11, float c12, float c21, float c22){
        vals[0] = c11;
        vals[1] = c12;
        vals[2] = c21;
        vals[3] = c22;
    }

    __device__
    Mat2(float4 v){
        vals[0] = v.x;
        vals[1] = v.y;
        vals[2] = v.z;
        vals[3] = v.w;
    }

    __device__
    float2 operator*(const float2& vec) {
        float2 res = make_float2(vals[0]*vec.x + vals[1]*vec.y,
                           vals[2]*vec.x + vals[3]*vec.y);

        return res;
    }

    __device__
    float operator[] (int index){
        return vals[index];
    }
};





__global__
void contourROIMean_kernel(float4* edges, ContourResult* results, cudaTextureObject_t depth_tex, int window_size){
    float4 edge = edges[threadIdx.x];
    ContourResult& result = results[threadIdx.x];

    /// Calculate Window region
    // get perpendicular vector
    float2 vec;
    vec.x = edge.x - edge.z;
    vec.y = edge.y - edge.w;
    float len = sqrtf(pow<float>(vec.x, 2) + pow<float>(vec.y, 2));
    float2 n_vec = make_float2(vec.x/len, vec.y/len);
    float angle = atan2(n_vec.y, n_vec.x) + M_PI / 2;

    Mat2 edge_space(make_float4(cos(angle), -sin(angle), sin(angle), cos(angle)));
    float2 tex_coord;
    int p_val_count = 0;
    int n_val_count = 0;
    int contour_len = 0;
    int n_sum = 0;
    int p_sum = 0;
    for(int i = 0; i < len; i++){
        ///--------- Perform for Edge Mean --------///
        // i,j coord in edge space
        tex_coord.x = 0;
        tex_coord.y = i;
        // convert to image space
        tex_coord = edge_space * tex_coord;
        // translate tex_coord
        tex_coord.x += edge.x;
        tex_coord.y += edge.y;
        // sample depth value at point
        int depth_val = (int)tex2D<float>(depth_tex, (int)tex_coord.y, (int)tex_coord.x);

        if(depth_val != 0){
            contour_len++;
            result.edge_mean += (int)depth_val;
        }

        for(int j = 0; j < window_size; j++){
            ///--------- Perform for P window --------///
            // i,j coord in edge space
            tex_coord.x = j;
            tex_coord.y = i;
            // convert to image space
            tex_coord = edge_space * tex_coord;
            // translate tex_coord
            tex_coord.x += edge.x;
            tex_coord.y += edge.y;
            // sample depth value at point
            int depth_val = (int)tex2D<float>(depth_tex, (int)tex_coord.y, (int)tex_coord.x);
            result.tval[j] = depth_val;

            if(depth_val != 0){
                p_val_count++;
                p_sum += (int)depth_val;
            }

            ///--------- Perform for N window --------///
            // i,j coord in edge space
            tex_coord.x = -1 * j;
            tex_coord.y = i;
            // convert to image space
            tex_coord = edge_space * tex_coord;
            // translate tex_coord
            tex_coord.x += edge.x;
            tex_coord.y += edge.y;
            // sample depth value at point
            depth_val = (int)tex2D<float>(depth_tex, (int)tex_coord.y, (int)tex_coord.x);

            if (depth_val > 0.001){
                n_val_count++;
                n_sum += (int)depth_val;
            }

        }
    }
    result.p_region_mean = p_sum / (float)p_val_count;
    result.p_region_count = p_val_count;
    result.n_region_mean = n_sum / (float)n_val_count;
    result.n_region_count = n_val_count;
    result.edge_mean /= contour_len;
    result.contour_len = contour_len;


}

__host__
void setupDepthMap(int width, int height, cv::Mat depth_map, cudaTextureObject_t& dev_depth_tex){
    size_t pitch;
    float* dev_dimg;
    uchar* dimg = depth_map.data;

    checkCudaErrors(cudaMallocPitch(&dev_dimg, &pitch, sizeof(float)*width, height));
    checkCudaErrors(cudaMemcpy2D(dev_dimg, pitch, dimg, depth_map.step, sizeof(float)*width,
            height, cudaMemcpyHostToDevice));

    /// Intialize Resoure Descriptor
    cudaResourceDesc texRes;
    memset(&texRes, 0x0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr   = dev_dimg;
    texRes.res.pitch2D.desc     = cudaCreateChannelDesc<float>();;
    texRes.res.pitch2D.width    = width;
    texRes.res.pitch2D.height   = height;
    texRes.res.pitch2D.pitchInBytes = pitch;

    /// Initialize Texture Descriptor
    cudaTextureDesc texDescr;
    memset(&texDescr, 0x0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&dev_depth_tex, &texRes, &texDescr, NULL));
}

__host__
ContourResult* launchContourRegionKernel(vector<vector<LineSegment>>& contour_segments, cv::Mat& depth_map, int window_size){
    float4* contours;
    float4* dev_contours;
    ContourResult* dev_contour_results;

    int edge_count = 0;
    for(auto& segments: contour_segments){
        edge_count += segments.size();
    }
    contours = (float4*)malloc(sizeof(float4) * edge_count);

    int t_edge_index = 0;
    for(auto& segments: contour_segments){
        for(auto& line: segments){
            contours[t_edge_index] = make_float4(line.getStartPos().x, line.getStartPos().y, line.getEndPos().x, line.getEndPos().y);
            t_edge_index += 1;
        }
    }
    checkCudaErrors(cudaMalloc((void**)&dev_contours, sizeof(float4) * edge_count));
    checkCudaErrors(cudaMalloc((void**)&dev_contour_results, sizeof(ContourResult) * edge_count));
    checkCudaErrors(cudaMemcpy(dev_contours, contours, sizeof(float4) * edge_count, cudaMemcpyHostToDevice));

    cudaTextureObject_t depth_tex;
    setupDepthMap(depth_map.rows, depth_map.cols, depth_map, depth_tex);

    /// Execute Cuda kernel
    contourROIMean_kernel<<<1, edge_count>>>(dev_contours, dev_contour_results, depth_tex, window_size);

    ContourResult* contour_results = (ContourResult*)malloc(sizeof(ContourResult) * edge_count);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(contour_results, dev_contour_results, sizeof(ContourResult) * edge_count, cudaMemcpyDeviceToHost);

    /// cleanup temp contours
    free(contours);
    // free(dev_contour_results);
    // free(dev_contours);

    return contour_results;
}

extern "C"
ContourResult* cu_determineROIMean(vector<vector<LineSegment>>& contour_segments, cv::Mat& depth_map, int window_size) {
    return launchContourRegionKernel(contour_segments, depth_map, window_size);
}

#endif //CUDA_CONTOUR_OPERATOINS_H