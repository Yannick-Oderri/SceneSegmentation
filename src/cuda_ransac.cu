#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>

#include "cuda.h"
#include "curand_kernel.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "cuda_ransac_kernel.h"

using namespace std;

// Calc the theoretical number of iterations using some conservative parameters
const double CONFIDENCE = 0.99;
const double INLIER_RATIO = 0.18; // Assuming lots of noise in the data!
const double INLIER_THRESHOLD = 3.0; // pixel distance
const int MAX_SUB_ITER = 256;

struct RansacParams{
    float threshold;
    float min_point_separation;
    float max_point_separation;
    int consensus_size;
};

struct RansacResults{
    int* valid_distances;
    int* valid_points;
};

__global__
void ransac_kernel(double3* consensus_set, RansacParams* ransac_params, int* result_matches, int* result_point_idx) {
    int exec_id = threadIdx.x;
    curandStateMRG32k3a rstate;
    curand_init(clock(), threadIdx.x, 0, &rstate);
    int sub_iter = 0;

    unsigned int iA, iB, iC;
    double3 A, B, C;
    double3 AB, AC;    
    double nAB, nAC;

    double distance;
    
    // find sampled point
    do {
        iA = curand(&rstate);
        iB = curand(&rstate);
        iC = curand(&rstate);
        iA = iA % ransac_params->consensus_size;
        iB = iB % ransac_params->consensus_size;
        iC = iC % ransac_params->consensus_size;

#if DEBUG == 1
        if(threadIdx.x == 0 )
            printf("A B C : %d %d %d\n", iA, iB, iC);
#endif
        A = consensus_set[iA];
        B = consensus_set[iB];
        C = consensus_set[iC];

        AB.x = B.x - A.x;
        AB.y = B.y - A.y;
        AB.z = B.z - A.z;

        AC.x = C.x - A.x;
        AC.y = C.y - A.y;
        AC.z = C.z - A.z;

        nAB = norm(3,(const double *) &AB);
        nAC = norm(3,(const double *) &AC);
        


#if DEBUG == 1
        if(threadIdx.x == 0 && blockIdx.x == 0) {
            printf("\nA : %lf %lf %lf\n", A.x, A.y, A.z);
            printf("B : %lf %lf %lf\n", B.x, B.y, B.z);
            printf("C : %lf %lf %lf\n", C.x, C.y, C.z);
            // printf("AB AC : %f %f\n", AB, AC );
        }
#endif
        if(nAC < ransac_params->min_point_separation || nAC < ransac_params->min_point_separation ||
            nAB < ransac_params->min_point_separation || nAB > ransac_params->max_point_separation ){
#if DEBUG == 1
            // if(threadIdx.x == 0 && blockIdx.x == 0)
            //     printf("\n\tAB, CA, Max pt separation %f %f %f\n", AB, CA, common_params->max_point_separation);// (float)(consensus_set_x[contour_id][iA] - consensus_set_x[contour_id][iB]), consensus_set_x[contour_id][iC] - consensus_set_x[contour_id][iA], common_params->max_point_separation);
#endif
            ++sub_iter;
        }else{
            break;
        }  
    } while (sub_iter <= MAX_SUB_ITER);


    if (sub_iter >= MAX_SUB_ITER)
    {
        return;
    }

    // calculate vector between points
    // AB.x = consensus_set[iA].x - consensus_set[iB].x;
    // AB.y = consensus_set[iA].y - consensus_set[iB].y;
    // AB.z = consensus_set[iA].z - consensus_set[iB].z;
    
    // AC.x = consensus_set[iA].x - consensus_set[iC].x;
    // AC.y = consensus_set[iA].y - consensus_set[iC].y;
    // AC.z = consensus_set[iA].z - consensus_set[iC].z;

    // calculate plane parameters 
    float alpha = ((AB.y*AC.z) - (AB.z*AC.z));
    float beta = ((AB.z*AC.x)-(AB.x*AC.z));
    float gamma = ((AB.x*AC.z)-(AB.y*AC.x));    
    float delta = 0;
    delta += A.x*((AB.z*AC.z)-(AB.y*AC.z));
    delta += A.y*((AB.x*AC.z)-(AB.z*AC.x));
    delta += A.z*((AB.y*AC.x)-(AB.x*AC.z));

    
    for (int i = 0; i < ransac_params->consensus_size; i++) {
        double3 point = consensus_set[i];
        distance = point.x * alpha + point.y * beta + point.z * gamma + delta;
        distance /= sqrtf((pow(alpha, 2) + pow(beta, 2), pow(gamma, 2)));
        distance = abs(distance);

        if (distance < ransac_params->threshold){
            result_matches[exec_id] += 1;
        }

    }

    result_point_idx[exec_id*3+0] = iA;
    result_point_idx[exec_id*3+1] = iB;
    result_point_idx[exec_id*3+2] = iC;
}

__host__
RansacResults* launch_ransac_kernel(int thread_count, double3* points, RansacParams* ransac_params) {
    RansacParams* dev_ransac_params;
    int* dev_results_match_count;
    int* dev_results_point_idx;

    RansacResults* host_results;
    double3* dev_points;

    printf("Allocating GPU Memory\n");
    // Ransac Params
    cudaMalloc((void **)&dev_ransac_params, sizeof(RansacParams));
    // Ransac Results    
    cudaMalloc((void**)&dev_results_match_count, sizeof(int) * ransac_params->consensus_size);
    cudaMalloc((void**)&dev_results_point_idx, sizeof(int) * ransac_params->consensus_size * 3);
    // Ransac Points
    cudaMalloc((void **)&dev_points, sizeof(double3) * ransac_params->consensus_size);
    
    // Copy Ransac Params from host to device
    cudaMemcpy(dev_ransac_params, ransac_params, sizeof(RansacParams), cudaMemcpyHostToDevice);
    // Copy ransac points from host tot device
    cudaMemcpy(dev_points, points, ransac_params->consensus_size * sizeof(double3), cudaMemcpyHostToDevice);

    // launch ransac kernel
    ransac_kernel<<<1, thread_count*2>>>(dev_points, dev_ransac_params, dev_results_match_count, dev_results_point_idx);
    cudaDeviceSynchronize();
    
    // generate ransac result  allocaiton
    host_results = (RansacResults*)malloc(sizeof(RansacResults));
    host_results->valid_distances = (int*)malloc(sizeof(int) * ransac_params->consensus_size);
    host_results->valid_points = (int*)malloc(sizeof(int) * ransac_params->consensus_size * 3);

    cudaMemcpy(host_results->valid_distances, dev_results_match_count, ransac_params->consensus_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_results->valid_points, dev_results_point_idx, ransac_params->consensus_size * sizeof(int) * 3, cudaMemcpyDeviceToHost);

    return host_results;
}

void readCSV(thrust::host_vector<double3> &data, const char *path)
{
    ifstream file(path);
    // thrust::host_vector<double3> data;

    string line;
    while (getline(file, line))
    {
        double3 p;
        sscanf(line.c_str(), "%lf,%lf,%lf", &p.x, &p.y, &p.z);
        data.push_back(p);
    }
    file.close();
}

extern "C" std::vector<double3> execute_ransac(std::vector<double3> &data) {
    RansacParams ransac_params;
    ransac_params.min_point_separation = 0.01f;
    ransac_params.max_point_separation = 1000.0f;
    ransac_params.threshold = 1.2;
    
    srand(420);

    double3* consensus_set;
    ransac_params.consensus_size = data.size();
    int datasize = data.size();
    

    // int ransac_iterations = 0;
    vector <double3> point_set;
    int total_points = point_set.capacity();
    const int thread_count = datasize;

    // Minimum Number of iterations to attempt
    float ransac_iterations = (log(1 - CONFIDENCE)/log(1 - pow(INLIER_RATIO, 3.0)));

    RansacResults* results = launch_ransac_kernel(thread_count, thrust::raw_pointer_cast(&data[0]), &ransac_params);

    int match_count = 0;
    int best_fit_idx = -1;
    for(int i = 0; i < datasize; i++){
        if (match_count < results->valid_distances[i]){
            match_count = results->valid_distances[i];
            best_fit_idx = i;
        }
    }

    printf("Best Match %d", best_fit_idx);
    printf("\nMatched Count: %d", match_count);
    std::vector<double3> valid_points(3);
    for(int i = 0; i < 3; i++){
        int point_idx = results->valid_points[best_fit_idx*3+i];
        printf("\nPoint %d: %lf, %lf, %lf", point_idx,
                                    data[point_idx].x,
                                    data[point_idx].y,
                                    data[point_idx].z);
        valid_points[i].x = data[point_idx].x;
        valid_points[i].y = data[point_idx].y;
        valid_points[i].z = data[point_idx].z;
    }
    return valid_points;
    
    // ransac_results

}