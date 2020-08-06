//
// Created by ynki9 on 5/25/20.
//

#include "cuda_depth_img_opr.h"
#include "component/cuda_depth_img_opr_impl.h"
#include <npp.h>
#include <math.h>
#include <thread>

__global__
void sobel_shared(cudaTextureObject_t src_img, cudaTextureObject_t preservation_map, Npp32f* x_grad, Npp32f* y_grad, Npp32s shared_pitch){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    Npp32f *y_row =
            (float *)(((char*)y_grad)+(y*shared_pitch));

    Npp32f *x_row =
            (float *)(((char*)x_grad)+(y*shared_pitch));

    //Npp32f y_kern[] = { -1, -2, -1, 0, 0, 0, 1, 2, 1};
    Npp32f y_kern[] = { 1, 0, -1, 2, 0, -2, 1, 0, -1};
    Npp32f y_conv_val = 0;
    // Perform gradient in y direction
    for(int u = 0; u<3; u++){
        int v = 0;
        int offset = -1;
        float sub_total = 0;
        while(v < 3){
            float val = tex2D<float>(src_img, x+u-1, y+v+offset);
            Npp8u preservation_val = tex2D<Npp8u>(preservation_map, x+u-1, y+v+offset);
            if(preservation_val != 0){
                if(v == 1){
//                    float val1 = tex2D<float>(src_img, x+u-1, y+v+offset)*y_kern[3];
//                    float val2 = tex2D<float>(src_img, x+u+1, y+v+offset)*y_kern[5];
//                    y_conv_val = val1 + val2;
                    u = 4;// If preserve center, preserve entire pixel
                    sub_total = 0;
                    y_conv_val = 0;
                    v = 4;
                    break;
                }else if(v == 2){
                    offset += -1; // If preserve bottom offset upwards
                }else{
                    offset += 1;  // If preserve top offset downwards
                }
                v = 0;
                sub_total = 0;
            }
            val = val * y_kern[u * 3 + v];
            sub_total += val;
            v += 1;
        }
        y_conv_val += sub_total;
    }


    Npp32f x_kern[] = { 1, 0, -1, 2, 0, -2, 1, 0, -1};
    Npp32f x_conv_val = 0;
    // Perform gradient in y direction
    for(int v = 0; v<3; v++){
        int u = 0;
        int offset = -1;
        float sub_total = 0;
        while(u < 3){
            float val = tex2D<float>(src_img, x+u+offset, y+v-1);
            Npp8u preservation_val = tex2D<Npp8u>(preservation_map, x+u+offset, y+v-1);
            if(preservation_val != 0){
                if(u == 1){
//                    float val1 = tex2D<float>(src_img, x+u+offset, y+v-1)*x_kern[3];
//                    float val2 = tex2D<float>(src_img, x+u+offset, y+v+1)*x_kern[5];
//                    x_conv_val = val1 + val2;
                    v = 4;// If preserve center, preserve entire pixel
                    x_conv_val = 0;
                    sub_total = 0;
                    u = 4;
                    break;
                }else if(u == 2){
                    offset += -1; // If preserve bottom offset upwards
                }else{
                    offset += 1;  // If preserve top offset downwards
                }
                u = 0;
                sub_total = 0;
            }
            val = val * x_kern[v * 3 + u];
            sub_total += val;
            u += 1;
        }
        x_conv_val += sub_total;
    }

    y_row[x] = y_conv_val;
    x_row[x] = x_conv_val;
}


__global__
void calculate_gvariance(cudaTextureObject_t src_img, NppiSize src_size, Npp32f* loc_mean,
                        Npp32f* loc_variance, Npp32s dst_step, float2* out_vals){
    __shared__ float2 cache[32*32];
    int cache_offset = threadIdx.x + blockDim.y * threadIdx.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    Npp32f* loc_mean_row = (Npp32f*)((Npp8u*)loc_mean + dst_step * y);
    Npp32f* loc_variance_row = (Npp32f*)((Npp8u*)loc_variance + dst_step * y);

    Npp32f local_mean = 0;
    Npp32f local_variance = 0;
    Npp32f inc_val = 0;

    Npp32f g = tex2D<Npp32f>(src_img, x, y);
    int wnd_size = 3;
    if(g > 1650 || g < 700){
        wnd_size += 14;
    }

    inc_val = 1;
    int wnd_range = wnd_size / 2;
    int valid_pix_count = 0;
    for (int i = 0; i < wnd_size * wnd_size; i++) {
        int u = i / wnd_size;
        int v = i % wnd_size;
        Npp32f pixel = tex2D<float>(src_img, (int) (x + v - wnd_range), (int) (y + u - wnd_range));
        valid_pix_count++;
        local_mean = local_mean + pixel;
        local_variance = local_variance + (pixel * pixel);
    }

    local_mean = local_mean / valid_pix_count;
    local_variance = local_variance / valid_pix_count;
    local_variance = local_variance - (local_mean * local_mean);

    // ensure local mean and variance are 0 for invalid regions
    loc_mean_row[x] = local_mean;
    loc_variance_row[x] = local_variance;

    cache[cache_offset].x = local_variance;
    cache[cache_offset].y = inc_val;

    __syncthreads();

    int i = blockDim.x * blockDim.y / 2;
    while(i != 0){
        if(cache_offset < i) {
            cache[cache_offset].x = cache[cache_offset].x + cache[cache_offset + i].x;
            cache[cache_offset].y = cache[cache_offset].y + cache[cache_offset + i].y;
        }

        __syncthreads();
        i /= 2;
    }

    if(cache_offset == 0)
        out_vals[blockIdx.x + blockDim.x * blockIdx.y] = cache[cache_offset];
}

__global__
void gwiener(cudaTextureObject_t src_img, NppiSize img_size,  cudaTextureObject_t src_variance,
            cudaTextureObject_t src_mean, Npp32f* dst_img, Npp32s dst_step, Npp32f noise){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    Npp32f* row = (Npp32f*)((Npp8u*)dst_img + dst_step * y);

    Npp32f g = tex2D<Npp32f>(src_img, x, y);
    Npp32f f = g;
    Npp32f local_variance = tex2D<Npp32f>(src_variance, x, y);
    Npp32f local_mean = tex2D<Npp32f>(src_mean, x, y);

    f = g - local_mean;
    g = local_variance - noise;
    g = fmaxf(g, 0);
    local_variance = fmaxf(local_variance, noise);
    f = f / local_variance;
    f = f * g;
    f = f + local_mean;
    row[x] = f;
}

__global__
void calculate_variance(cudaTextureObject_t src_img, NppiSize src_size, Npp32f* loc_mean,
                        Npp32f* loc_variance, Npp32s dst_step, Npp32u wnd_size, float2* out_vals){
    __shared__ float2 cache[32*32];
    int cache_offset = threadIdx.x + blockDim.y * threadIdx.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    Npp32f* loc_mean_row = (Npp32f*)((Npp8u*)loc_mean + dst_step * y);
    Npp32f* loc_variance_row = (Npp32f*)((Npp8u*)loc_variance + dst_step * y);

    Npp32f local_mean = 0;
    Npp32f local_variance = 0;
    Npp32f inc_val = 0;

    Npp32f g = tex2D<Npp32f>(src_img, x, y);
    if(g != 0) { // do not perform operation on invalid region
        inc_val = 1;
        int wnd_range = wnd_size / 2;
        int valid_pix_count = 0;
        for (int i = 0; i < wnd_size * wnd_size; i++) {
            int u = i / wnd_size;
            int v = i % wnd_size;
            Npp32f pixel = tex2D<float>(src_img, (int) (x + v - wnd_range), (int) (y + u - wnd_range));
            if (pixel != 0) { // avoid invalid regions
                valid_pix_count++;
                local_mean = local_mean + pixel;
                local_variance = local_variance + (pixel * pixel);
            }
        }


        local_mean = local_mean / valid_pix_count;
        local_variance = local_variance / valid_pix_count;
        local_variance = local_variance - (local_mean * local_mean);

    }

    // ensure local mean and variance are 0 for invalid regions
    loc_mean_row[x] = local_mean;
    loc_variance_row[x] = local_variance;

    cache[cache_offset].x = local_variance;
    cache[cache_offset].y = inc_val;

    __syncthreads();

    int i = blockDim.x * blockDim.y / 2;
    while(i != 0){
        if(cache_offset < i) {
            cache[cache_offset].x = cache[cache_offset].x + cache[cache_offset + i].x;
            cache[cache_offset].y = cache[cache_offset].y + cache[cache_offset + i].y;
        }

        __syncthreads();
        i /= 2;
    }

    if(cache_offset == 0)
        out_vals[blockIdx.x + blockDim.x * blockIdx.y] = cache[cache_offset];
}

__global__
void calculate_variance_32f4c(cudaTextureObject_t src_img, NppiSize src_size, Npp32f* loc_mean,
                        Npp32f* loc_variance, Npp32s dst_step, Npp32u wnd_size, float2* out_vals){
    __shared__ float2 cache[32*32*4];
    int cache_offset = threadIdx.x + blockDim.y * threadIdx.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int c = blockIdx.z;

    Npp32f* loc_mean_row = ((Npp32f*)((Npp8u*)loc_mean + dst_step * y));
    Npp32f* loc_variance_row = ((Npp32f*)((Npp8u*)loc_variance + dst_step * y));

    Npp32f local_mean = 0;
    Npp32f local_variance = 0;
    Npp32f inc_val = 0;

    float4 pix = tex2D<float4>(src_img, x, y);
    float* g = &pix.x;

    if(g[c] != 0) { // do not perform operation on invalid region
        inc_val = 1;
        int wnd_range = wnd_size / 2;
        int valid_pix_count = 0;
        for (int i = 0; i < wnd_size * wnd_size; i++) {
            int u = i / wnd_size;
            int v = i % wnd_size;
            float4 tpixel = tex2D<float4>(src_img, (int) (x + v - wnd_range), (int) (y + u - wnd_range));
            float* pixel = &tpixel.x;
            if (pixel[c] != 0) { // avoid invalid regions
                valid_pix_count++;
                local_mean = local_mean + pixel[c];
                local_variance = local_variance + (pixel[c] * pixel[c]);
            }
        }


        local_mean = local_mean / valid_pix_count;
        local_variance = local_variance / valid_pix_count;
        local_variance = local_variance - (local_mean * local_mean);

    }

    // ensure local mean and variance are 0 for invalid regions
    loc_mean_row[x*4+c] = local_mean;
    loc_variance_row[x*4+c] = local_variance;

    cache[cache_offset].x = local_variance;
    cache[cache_offset].y = inc_val;

    __syncthreads();

    int i = blockDim.x * blockDim.y / 2;
    while(i != 0){
        if(cache_offset < i) {
            cache[cache_offset].x = cache[cache_offset].x + cache[cache_offset + i].x;
            cache[cache_offset].y = cache[cache_offset].y + cache[cache_offset + i].y;
        }

        __syncthreads();
        i /= 2;
    }

    if(cache_offset == 0)
        out_vals[blockIdx.x + blockDim.x * blockIdx.y + blockDim.x * blockDim.y * blockIdx.z] = cache[cache_offset];
}

__global__
void wiener_32f4c(cudaTextureObject_t src_img, NppiSize img_size,  cudaTextureObject_t src_variance,
            cudaTextureObject_t src_mean, Npp32f* dst_img, Npp32s dst_step, float4 tnoise){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int c = blockIdx.z;

    Npp32f* row = (Npp32f*)((Npp8u*)dst_img + dst_step * y);

    float4 vpix = tex2D<float4>(src_img, x, y);
    float* pg = &vpix.x;
    Npp32f f = pg[c];
    float g = pg[c];
    float* pnoise = &tnoise.x;
    float noise = pnoise[c];
    if(g != 0) {
        float4 tlocal_variance = tex2D<float4>(src_variance, x, y);
        float* plocal_variance = &tlocal_variance.x;
        float local_variance = plocal_variance[c];
        float4 tlocal_mean = tex2D<float4>(src_mean, x, y);
        float* plocal_mean = &tlocal_mean.x;
        float local_mean = plocal_mean[c];

        f = g - local_mean;
        g = local_variance - noise;
        g = fmaxf(g, 0);
        local_variance = fmaxf(local_variance, noise);
        f = f / local_variance;
        f = f * g;
        f = f + local_mean;
    }
    row[x*4+c] = f;
}
__global__
void wiener(cudaTextureObject_t src_img, NppiSize img_size,  cudaTextureObject_t src_variance,
            cudaTextureObject_t src_mean, Npp32f* dst_img, Npp32s dst_step, Npp32f noise){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    Npp32f* row = (Npp32f*)((Npp8u*)dst_img + dst_step * y);

    Npp32f g = tex2D<Npp32f>(src_img, x, y);
    Npp32f f = g;
    if(g != 0) {
        Npp32f local_variance = tex2D<Npp32f>(src_variance, x, y);
        Npp32f local_mean = tex2D<Npp32f>(src_mean, x, y);

        f = g - local_mean;
        g = local_variance - noise;
        g = fmaxf(g, 0);
        local_variance = fmaxf(local_variance, noise);
        f = f / local_variance;
        f = f * g;
        f = f + local_mean;
    }
    row[x] = f;
}

__host__
void wiener_opr(cudaTextureObject_t src_tex, NppiSize src_size,
                cudaTextureObject_t variance_tex, Npp32f* variance_buffer,
                cudaTextureObject_t mean_tex, Npp32f* mean_buffer,
                Npp32f* output_buffer, Npp32s buffer_step, Npp32s roi_size){
    NppiSize target_thread = {32, 32};
    int block_width = src_size.width/target_thread.width;
    int block_height = src_size.height / target_thread.height;
    int block_size = block_width * block_height;

    static float2* host_buffer;
    if (host_buffer == nullptr){
        host_buffer = (float2*)malloc(sizeof(float2)*block_size);
    }

     dim3 blocks(block_width, block_height);
    dim3 threads(target_thread.width, target_thread.height);
    float2* gv_buffer = (float2*)output_buffer;
    // Calculate per-pixel variance, per-pixel mean, and global variance
    if(roi_size < 1)
        calculate_gvariance<<<blocks, threads>>>(src_tex, src_size, mean_buffer, variance_buffer, buffer_step, gv_buffer);
    else
        calculate_variance<<<blocks, threads>>>(src_tex, src_size, mean_buffer, variance_buffer, buffer_step, roi_size, gv_buffer);

    // Global variance needs to be calculated per block
    Npp32f global_variance = 0;
    Npp32f variance_count = 0;
    cudaMemcpy(host_buffer, gv_buffer, sizeof(float2)*block_size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < block_size; i++){
        global_variance += host_buffer[i].x;
        variance_count += host_buffer[i].y;
    }
    global_variance /= variance_count;

    // Perform wiener filtering
    if(roi_size < 1)
        gwiener<<<blocks, threads>>>(src_tex, src_size, variance_tex, mean_tex, output_buffer, buffer_step, global_variance);
    else
        wiener<<<blocks, threads>>>(src_tex, src_size, variance_tex, mean_tex, output_buffer, buffer_step, global_variance);
}


__host__
void wiener_opr(CurveDiscTextureUnits& tunit, TexID_F32 src_id, TexID_F32 dst_id, int filter_size){
    wiener_opr(tunit.getTexture(src_id), {tunit.getWidth(), tunit.getHeight()},
               tunit.getTexture(SBUFFER5_F32), tunit.getBuffer(SBUFFER5_F32),
               tunit.getTexture(SBUFFER6_F32), tunit.getBuffer(SBUFFER6_F32),
               tunit.getBuffer(dst_id), tunit.f32Pitch(), filter_size);
}


__host__
void wiener_opr(CurveDiscTextureUnits& tunit, TexID_F32C4 src_id, TexID_F32C4 dst_id, int filter_size){
    NppiSize src_size = {tunit.getWidth(), tunit.getHeight()};
    cudaTextureObject_t src_tex = tunit.getTexture(src_id);
    cudaTextureObject_t variance_tex = tunit.getTexture(SBUFFER2_F32C4);
    Npp32f* variance_buffer = (float*)tunit.getBuffer(SBUFFER2_F32C4);
    cudaTextureObject_t mean_tex = tunit.getTexture(SBUFFER3_F32C4);
    Npp32f* mean_buffer = (float*)tunit.getBuffer(SBUFFER3_F32C4);
    Npp32f* output_buffer = (float*)tunit.getBuffer(dst_id);
    int buffer_step = tunit.f32c4Pitch();
    int roi_size = filter_size;

    NppiSize target_thread = {32, 32};
    int block_width = src_size.width/target_thread.width;
    int block_height = src_size.height / target_thread.height;
    int block_size = block_width * block_height;

    static float2* host_buffer;
    if (host_buffer == nullptr){
        host_buffer = (float2*)malloc(sizeof(float2)*block_size*4);
    }

    dim3 blocks(block_width, block_height, 4);
    dim3 threads(target_thread.width, target_thread.height);
    float2* gv_buffer = (float2*)output_buffer;
    // Calculate per-pixel variance, per-pixel mean, and global variance
    calculate_variance_32f4c<<<blocks, threads>>>(src_tex, src_size, mean_buffer, variance_buffer, buffer_step, roi_size, gv_buffer);

    // Global variance needs to be calculated per block
    float4 tglobal_variance = {0, 0, 0, 0};
    float* global_variance = &tglobal_variance.x;
    float4 tvariance_count = {0, 0, 0, 0};
    float* variance_count = &tvariance_count.x;
    cudaMemcpy(host_buffer, gv_buffer, sizeof(float2)*block_size*4, cudaMemcpyDeviceToHost);
    for(int j = 0; j < 4; j++) {
        for (int i = 0; i < block_size; i++) {
            global_variance[j] += host_buffer[j*block_size+i].x;
            variance_count[j] += host_buffer[j*block_size+i].y;
        }
        global_variance[j] /= variance_count[j];
    }

    // Perform wiener filtering
    wiener_32f4c<<<blocks, threads>>>(src_tex, src_size, variance_tex, mean_tex, output_buffer, buffer_step, tglobal_variance);
}

__global__
void wiener2(cudaTextureObject_t src_img, NppiSize src_size, Npp32f* dst_img, Npp32s dst_step, Npp32f noise){
    Npp32f *row =
            (float *)(((char*)dst_img)+(blockIdx.x*dst_step));

    int l = src_size.width / blockDim.x;
    int b = threadIdx.x * l;

    for (int i = b; i < b + l; i++){
        int x = i;
        int y = blockIdx.x;
        Npp32f g = tex2D<float>(src_img, (int) i, (int) blockIdx.x);
        if(g==0){ // If pixel is 0, set pixel to closest valid value
            Npp32f col = 0;
            for(int j = 0; j < 9; j++){
                int u = j / 3;
                int v = j % 3;
                g = tex2D<Npp32f>(src_img, u+x-1, v+y-1);
                if(g != 0) {
                    col = g;
                    continue;
                }
            }

            row[i] = col;
            continue;
        }

        Npp32f local_mean = 0;
        Npp32f local_variance = 0;
        int wnd_size =  5;
        int wnd_range = wnd_size / 2;
        int valid_pix_count = 0;
        for(int u = 0; u < wnd_size; u++){
            for(int v = 0; v < wnd_size; v++){
                Npp32f pixel = tex2D<float>(src_img, (int) (i+v-wnd_range), (int)(blockIdx.x+u-wnd_range));
                valid_pix_count++;
                local_mean = local_mean + pixel;
                local_variance = local_variance + (pixel * pixel);
            }
        }

        local_mean = local_mean / valid_pix_count; // (wnd_size*wnd_size);
        local_variance = local_variance / valid_pix_count; //(wnd_size*wnd_size);
        local_variance = local_variance - (local_mean * local_mean);

        Npp32f f = g - local_mean;
        g = local_variance - noise;
        g = fmaxf(g, 0);
        local_variance = fmaxf(local_variance, noise);
        f = f / local_variance;
        f = f * g;
        f = f + local_mean;

        row[i] = f;
    }
}



__global__
void gradientOperation(Npp32f* x_grad, Npp32f* y_grad,
        NppiSize src_size, Npp32f* lr_img, Npp32f* ud_img, Npp32s dst_step, Npp32f grad_tresh){
    Npp32f *x_grad_row =
            (float *)(((char*)x_grad)+(blockIdx.x*dst_step));
    Npp32f *y_grad_row =
            (float *)(((char*)y_grad)+(blockIdx.x*dst_step));

    Npp32f *ud_gdir =
            (float *)(((char*)ud_img)+(blockIdx.x*dst_step));
    Npp32f *lr_gdir =
            (float *)(((char*)lr_img)+(blockIdx.x*dst_step));

    int l = src_size.width / blockDim.x;
    int b = threadIdx.x * l;

    for (int i = b; i < b + l; i++) {
        float x_grad = x_grad_row[i]; //tex2D<float>(x_grad, (float) i, (float) blockIdx.x);
        float y_grad = y_grad_row[i]; //tex2D<float>(y_grad, (float) i, (float) blockIdx.x);
        if (abs(x_grad) < grad_tresh && abs(y_grad) < grad_tresh){
            ud_gdir[i] = 0;
            lr_gdir[i] = 0;
            continue;
        }

        float gdir = atan2(-y_grad, x_grad);
        lr_gdir[i] =  gdir;
        ud_gdir[i] = abs(gdir);
        if (gdir > M_PI/2) {
            lr_gdir[i] = M_PI - gdir;
        }else if(gdir < -M_PI/2) {
            lr_gdir[i] = -(M_PI+gdir);
        }
    }
}


__global__
void gradientFollow(cudaTextureObject_t xgrad, cudaTextureObject_t ygrad, Npp32f* src_img, Npp32f* dst_img, int pitch){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //float* row = (float*)((unsigned char*)dst_img + y*pitch);
    float* src_row = (float*)((unsigned char*)src_img+ y*pitch);

    float2 vec;
    vec.x = tex2D<float>(xgrad, x, y);
    vec.y = tex2D<float>(ygrad, x, y);
    float mag = sqrt(pow(vec.x, 2)+pow(vec.y, 2));
    float src_val = src_row[x]; //tex2D<float>(src_img, x, y);
    if(src_val > 1) {
        float *new_row = (float *) ((unsigned char *) dst_img + (int)(y+vec.y/mag) * pitch);

        new_row[(int)(x+vec.x/mag)] = src_val;
        src_row[x] = 0;
    }
}


__global__
void gradient_normals(cudaTextureObject_t src_img, NppiSize src_size, cudaTextureObject_t xgrad,
                      cudaTextureObject_t ygrad, float4* dst_img, Npp32u dst_pitch){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float4* row = (float4*)((char*)(dst_img) + y * dst_pitch);

    Npp32f pix = tex2D<Npp32f>(src_img, x, y);
    Npp32f dzdx = tex2D<Npp32f>(xgrad, x, y) / 2.0;
    Npp32f dzdy = tex2D<Npp32f>(ygrad, x, y) / 2.0;

    float4 n = {0, 0, 0, 0};
    if(pix != 0){
        float3 d = {-dzdx, -dzdy, 1.0};
        float mag = sqrt(pow(d.x, 2) + pow(d.y, 2) + pow(d.z, 2)); // norm3df(d.x, d.y, d.z);
        n.x = ((d.x / mag)) ;
        n.y = ((d.y / mag)) ;
        n.z = ((d.z / mag)) ;
        n.w = mag;
    }
    row[x] = n;
}

__global__
void normal_vector_seperation(cudaTextureObject_t normal_img, NppiSize src_size, Npp32f* x_img, Npp32f* y_img, Npp32u dst_pitch){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    float* xrow = (float*)((char*)(x_img) + y * dst_pitch);
    float* yrow = (float*)((char*)(y_img) + y * dst_pitch);

    xrow[x] = 0;
    yrow[x] = 0;
    float4 vec = tex2D<float4>(normal_img, x, y);
    float mag = sqrt(pow(vec.x, 2) + pow(vec.y, 2) + pow(vec.z, 2)); //norm3df(vec.x, vec.y, vec.z);
    vec.x /= mag;
    vec.y /= mag;
    vec.z /= mag;

    float4 norm;
    float sv[] = {1, 1, 1};
    float hori_val = 0;
    float vert_val = 0;
    for(int v=0; v < 3; v++) {
        float4 tres1 = tex2D<float4>(normal_img, x-1, y+v-1);
        float mag_1 = sqrt(pow(tres1.x, 2) + pow(tres1.y, 2) + pow(tres1.z, 2));
        tres1.x /= mag_1;
        tres1.y /= mag_1;
        tres1.z /= mag_1;

        float4 tres2 = tex2D<float4>(normal_img, x+2, y+v-1);
        float mag_2 = sqrt(pow(tres2.x, 2) + pow(tres2.y, 2) + pow(tres2.z, 2));
        tres2.x /= mag_2;
        tres2.y /= mag_2;
        tres2.z /= mag_2;

        float val1 = pow(tres1.x * vec.x + tres1.y * vec.y + tres1.z * vec.z, 3);
        float val2 = pow(tres2.x * vec.x + tres2.y * vec.y + tres2.z * vec.z, 3);
        float mean = (val1+val2)/2;
        vert_val += (1-mean)*sv[v]*500;


        tres1 = tex2D<float4>(normal_img, x+v-1, y-1);
        mag_1 = sqrt(pow(tres1.x, 2) + pow(tres1.y, 2) + pow(tres1.z, 2));
        tres1.x /= mag_1;
        tres1.y /= mag_1;
        tres1.z /= mag_1;

        tres2 = tex2D<float4>(normal_img, x+v-1, y+1);
        mag_2 = sqrt(pow(tres2.x, 2) + pow(tres2.y, 2) + pow(tres2.z, 2));
        tres2.x /= mag_2;
        tres2.y /= mag_2;
        tres2.z /= mag_2;


        val1 = pow(tres1.x*vec.x + tres1.y*vec.y + tres1.z * vec.z, 3);
        val2 = pow(tres2.x*vec.x + tres2.y*vec.y + tres2.z * vec.z, 3);
        mean = (val1+val2)/2;
        hori_val += (1-mean)*sv[v]*500;
    }


    if(abs(hori_val/3) > 11.0)
        xrow[x] = 100;

    if(abs(vert_val / 5.0) > 11.0)
        yrow[x] = 100;

}



__global__
void determineTextureRange(cudaTextureObject_t src_img, Npp32f* out_vals){
    __shared__ float cache[32*32];
    int cache_offset = threadIdx.x + blockDim.y * threadIdx.y;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    cache[cache_offset] = tex2D<float>(src_img, x, y);
    __syncthreads();

    int i = blockDim.x * blockDim.y / 2;
    while(i != 0){
        if(cache_offset < i)
            cache[cache_offset] = fmaxf(cache[cache_offset], cache[cache_offset + i]);

        __syncthreads();
        i /= 2;
    }

    if(cache_offset == 0)
        out_vals[blockIdx.x + blockDim.x * blockIdx.y] = cache[cache_offset];
}

__host__
void determineTextureRange(cudaTextureObject_t tex_obj, NppiSize img_size, Npp32f& max_val, Npp32f& min_val){
    NppiSize target_thread = {32, 32};
    int block_width = img_size.width/target_thread.width;
    int block_height = img_size.height / target_thread.height;
    int block_size = block_width * block_height;

    //Create Return buffer
    Npp32f* dev_buffer;
    cudaMalloc(&dev_buffer, sizeof(float) * block_size);

    dim3 blocks(block_width, block_height);
    dim3 threads(target_thread.width, target_thread.height);

    determineTextureRange<<<blocks, threads>>>(tex_obj, dev_buffer);
    checkCudaErrors(cudaDeviceSynchronize());

    Npp32f* host_buffer = static_cast<Npp32f *>(malloc(sizeof(float) * block_size));
    cudaMemcpy(host_buffer, dev_buffer, sizeof(float)*block_size, cudaMemcpyDeviceToHost);
    max_val = 0;
    min_val = host_buffer[0];
    for(int i = 0; i < block_size; i++){
        max_val = std::max(max_val, host_buffer[i]);
        min_val = std::min(min_val, host_buffer[i]);
    }

    free(host_buffer);
    cudaFree(dev_buffer);

}



__global__
void convert_32f_8u_scaled(cudaTextureObject_t tex_obj, cudaTextureObject_t tex_depth, Npp8u* dst_img, int dst_step, Npp32f alpha, Npp32f beta){
    int x = (threadIdx.x * 4) + blockIdx.x * blockDim.x * 4;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    Npp32f gamma = beta - alpha;

    Npp32u *row =
            (Npp32u *)(((Npp8u*)dst_img)+(y*dst_step));
    Npp32u final_val = 0;
    for(int i = 0; i < 4; i++) {
        Npp32f val = tex2D<Npp32f>(tex_depth, x+i, y);
        if(val != 0)
            val = tex2D<Npp32f>(tex_obj, x+i, y);
        Npp32u cval = (Npp8u)((val-alpha)/gamma * 255);
        final_val |= (cval << (8*i));
    }
    row[x/4] = final_val;

}

__global__
void convert_8u_32f_scaled(cudaTextureObject_t tex_obj, Npp32f* dst_img, int dst_step, Npp32f alpha, Npp32f beta){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    Npp32f *row =
            (Npp32f *)(((Npp8u*)dst_img)+(y*dst_step));
    Npp32f final_val = 0;
    Npp32u val = (tex2D<Npp8u>(tex_obj, x, y)); // >> (x % 4)) & 0xff;

    final_val = __uint2float_rd(val);
    final_val = (final_val); // + alpha) * beta;

    row[x] = final_val;
}

__global__
void or_8u(cudaTextureObject_t tex1, cudaTextureObject_t tex2, Npp8u* dst_img, int dst_step){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    Npp8u *row =
            (Npp8u *)(((Npp8u*)dst_img)+(y*dst_step));
    Npp8u val1 = tex2D<Npp8u>(tex1, x, y);
    Npp8u  val2 = tex2D<Npp8u>(tex2, x, y);

    row[x] = (val1 | val2);
}

__global__
void xor_8u(cudaTextureObject_t tex1, cudaTextureObject_t tex2, Npp8u* dst_img, int dst_step){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    Npp8u *row =
            (Npp8u *)(((Npp8u*)dst_img)+(y*dst_step));
    Npp8u val1 = tex2D<Npp8u>(tex1, x, y);
    Npp8u  val2 = tex2D<Npp8u>(tex2, x, y);

    row[x] = (val1 ^ val2);
}

__global__
void xor_8u_step(cudaTextureObject_t tex1, cudaTextureObject_t tex2, Npp8u* dst_img, int dst_step, bool x_even_step, bool y_even_step){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if((x % 2) == x_even_step)
        return;
    if((y % 2) == y_even_step)
        return;

    Npp8u *row =
            (Npp8u *)(((Npp8u*)dst_img)+(y*dst_step));
    Npp8u val1 = tex2D<Npp8u>(tex1, x, y);
    Npp8u  val2 = tex2D<Npp8u>(tex2, x, y);

    row[x] = (val1 ^ val2);
}

__global__
void and_8u(cudaTextureObject_t tex1, cudaTextureObject_t tex2, Npp8u* dst_img, int dst_step){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    Npp8u *row =
            (Npp8u *)(((Npp8u*)dst_img)+(y*dst_step));
    Npp8u val1 = tex2D<Npp8u>(tex1, x, y);
    Npp8u  val2 = tex2D<Npp8u>(tex2, x, y);

    row[x] = (val1 & val2);
}

__global__
void interpolate_invalid_regions(cudaTextureObject_t src_img, Npp32f* dst_img, int dst_step){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    Npp32f* row = (Npp32f*)((Npp8u *)dst_img+(y * dst_step));

    Npp32f pix_val = tex2D<Npp32f>(src_img, x, y);
    if (pix_val != 0){
        row[x] = pix_val;
        return;
    }
    for(int v = y-1; v <= y+1; v++){
        for(int u = (x-1); u <= (x+1); u++){
            pix_val = tex2D<Npp32f>(src_img, u, v);
            if(pix_val != 0)
                row[x] = pix_val;
            return;
        }
    }
    row[x] = 0;
}

__global__
void interpolate_invalid_regions_8u(cudaTextureObject_t src_img, Npp8u* dst_img, int dst_step){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    Npp8u* row = (Npp8u*)((Npp8u *)dst_img+(y * dst_step));

    Npp8u pix_val = tex2D<Npp8u>(src_img, x, y);
    if (pix_val > 100){
        row[x] = 255;
        return;
    }
    for(int v = y-1; v <= y+1; v++){
        for(int u = (x-1); u <= (x+1); u++){
            pix_val = tex2D<Npp8u>(src_img, u, v);
            if(pix_val > 100)
                row[x] = 255;
            return;
        }
    }
    row[x] = 0;
}

__global__
void applyLUT(cudaTextureObject_t src_img, Npp8u* dst_img, int dst_step, int lut_offset=0){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned char *row =
            (unsigned char *)(((char*)dst_img)+(y*dst_step));

    int L = 0;
    int k=1;
    for ( int j=0; j<3; ++j)
    {
        for ( int u=0;u<3;++u )
        {
            if (tex2D<unsigned char>(src_img, x+u-1, y+j-1) != 0)
                L += (k<<u);
        }
        k<<=3;
    }
    row[x] = DEV_LUT[L + lut_offset];
}

__global__
void applyLUT_inv(cudaTextureObject_t src_img, Npp8u* dst_img, int dst_step, int lut_offset=0){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned char *row =
            (unsigned char *)(((char*)dst_img)+(y*dst_step));

    int L = 0;
    int k=1;
    for ( int j=0; j<3; ++j)
    {
        for ( int u=0;u<3;++u )
        {
            if (tex2D<unsigned char>(src_img, x+u-1, y+j-1) == 0)
                L += (k<<u);
        }
        k<<=3;
    }
    row[x] = DEV_LUT[L + lut_offset];
}


__global__
void gradientFollow2(cudaTextureObject_t xgrad, cudaTextureObject_t ygrad, cudaTextureObject_t end_points, Npp8u* dst_img, int pitch, int len){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned char *row =
            (unsigned char *)(((char*)dst_img)+(y*pitch));

    unsigned char endpoint_val = tex2D<unsigned char>(end_points, x, y);
    float current_len = 1;
    if(endpoint_val > 0){ // if valid endpoint
        float2 vec; // Gradient vector
        vec.x = tex2D<float>(xgrad, x, y);
        vec.y = tex2D<float>(ygrad, x, y);
        float mag = sqrt(pow(vec.x, 2)+pow(vec.y, 2)); // Magnitude

        for(int i = 0; i < len; i++){
            x = (int)(x+vec.x/mag*current_len);
            y = (int)(y+vec.y/mag*current_len);
            unsigned char *new_row = (unsigned char *) ((unsigned char *) dst_img + y * pitch);
            new_row[x] = 0x255;

            vec.x = tex2D<float>(xgrad, x, y);
            vec.y = tex2D<float>(ygrad, x, y);
            mag = sqrt(pow(vec.x, 2)+pow(vec.y, 2)); // Magnitude
            current_len += 0.95;
        }
    }
}

__constant__
float* CUDA_AVG_FRAME_LIST[MAX_FRAME_TRACKING];

__global__
void average_frame_buffers_f32(int frame_list_size, Npp32f* dst_img, int pitch){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float total = 0;
    float valid_count = 0;


    for(int i = 0; i < frame_list_size; i++){
        float* frame_buffer = CUDA_AVG_FRAME_LIST[i];
        float *trow =
                (float *)(((char*)frame_buffer)+(y*pitch));
        int val = trow[x];
        total += val;
        if(val > 0)
            valid_count = valid_count + 1;
    }
    float *row =
            (float *)(((char*)dst_img)+(y*pitch));
    valid_count = valid_count == 0? 1: valid_count;
    row[x] = total/valid_count;
}

__host__
void getRunningAverage_f32(CurveDiscTextureUnits& tunits, Npp32f* const frame_buffers[], int frame_len,
        int front, TexID_F32 new_frame, TexID_F32 dst_id){
    // Append new frame to circular buffer
    int new_front = (front+1) % frame_len;
    // Update Front index
    front = new_front;
    // Update Frame
    checkCudaErrors(cudaMemcpy(frame_buffers[new_front], tunits.getBuffer(new_frame),
            tunits.f32Pitch()*tunits.getHeight(), cudaMemcpyDeviceToDevice));

    // BUffer infor
    static Npp32f** host_buffer;
    if(host_buffer == nullptr){
        host_buffer = (Npp32f**)malloc(sizeof(Npp32f *) * frame_len);
    }
//    static Npp32f** dev_buffer;
//    if(dev_buffer == nullptr){
//        cudaMalloc(&dev_buffer, sizeof(Npp32f*) * frame_len);
//    }

    // Normalize frame list to front being 0
    for(int i = new_front, j=0; i > (new_front - frame_len); i--, j++){
        int frame_idx = i % frame_len;
        host_buffer[j] = frame_buffers[frame_idx];
    }
    // Copy frame list to device
    cudaMemcpyToSymbol(CUDA_AVG_FRAME_LIST, host_buffer, sizeof(float*)*frame_len);
    //checkCudaErrors(cudaMemcpy(dev_buffer, host_buffer, sizeof(Npp32f*)*frame_len, cudaMemcpyHostToDevice));

    // Cuda Threading info
    NppiSize target_thread = {32, 32};
    int block_width = tunits.getWidth() / target_thread.width;
    int block_height = tunits.getHeight() / target_thread.height;
    int block_size = block_width * block_height;
    dim3 blocks(block_width, block_height);
    dim3 threads(target_thread.width, target_thread.height);

    // Perform Averaging
    //average_frame_buffers_f32<<<blocks, threads>>>(frame_len, tunits.getBuffer(dst_id), tunits.f32Pitch());

}

float** depth_frame_buffers = nullptr;
__host__
void averageDepthFrame(CurveDiscTextureUnits& tunits, TexID_F32 frame_id, TexID_F32 result_id){
    int buffer_len = 5; // AVG_DEPTH_FRAMES_LEN;
    if(depth_frame_buffers == nullptr){ // Alocate depth frame buffer if not available
        depth_frame_buffers = (Npp32f**)malloc(sizeof(Npp32f*)*buffer_len);
        size_t pitch;
        for(int i = 0; i < buffer_len; i++){
            checkCudaErrors(cudaMallocPitch(&depth_frame_buffers[i], &pitch, sizeof(float)*tunits.getWidth(), tunits.getHeight()));
        }
    }else{
        int x = 5;
    }

    static int front = 0;

    getRunningAverage_f32(tunits, depth_frame_buffers, buffer_len, front, frame_id, result_id);
    front++;
}

__host__
void despeckle(CurveDiscTextureUnits& tunits, TexID_U8 src_id, TexID_U8 dst_id, int n) {
    cudaMemcpyToSymbol(DEV_LUT, LUT_spur1, sizeof(unsigned char)*512);

    NppiSize target_thread = {32, 32};
    int block_width = tunits.getWidth() / target_thread.width;
    int block_height = tunits.getHeight() / target_thread.height;
    int block_size = block_width * block_height;
    dim3 blocks(block_width, block_height);
    dim3 threads(target_thread.width, target_thread.height);

    applyLUT<<<blocks, threads>>>(
            tunits.getTexture(src_id),
            tunits.getBuffer(dst_id), tunits.u8Pitch(), 0);

}

__host__
void fatten(CurveDiscTextureUnits& tunits, TexID_U8 src_id, TexID_U8 dst_id, int n) {
    cudaMemcpyToSymbol(DEV_LUT, LUT_fatten, sizeof(unsigned char)*512);

    NppiSize target_thread = {32, 32};
    int block_width = tunits.getWidth() / target_thread.width;
    int block_height = tunits.getHeight() / target_thread.height;
    int block_size = block_width * block_height;
    dim3 blocks(block_width, block_height);
    dim3 threads(target_thread.width, target_thread.height);

    TexID_U8 tsid = src_id;
    TexID_U8 tmid1 = SBUFFER2_U8;
    cudaMemcpy(tunits.getBuffer(tmid1),
            tunits.getBuffer(src_id),
            tunits.u8Pitch()*tunits.getHeight(),
            cudaMemcpyDeviceToDevice);
    TexID_U8 tmid2 = SBUFFER3_U8;
    for(int i =0; i < n; i++) {
        applyLUT<<<blocks, threads>>>(
                tunits.getTexture(tmid1),
                tunits.getBuffer(tmid2), tunits.u8Pitch(), 0);
        TexID_U8 t = tmid1;
        tmid1 = tmid2;
        tmid2 = t;
    }

    cudaMemcpy(tunits.getBuffer(dst_id),
               tunits.getBuffer(tmid2),
               tunits.u8Pitch()*tunits.getHeight(),
               cudaMemcpyDeviceToDevice);

}


__host__
void majority(CurveDiscTextureUnits& tunits, TexID_U8 src_id, TexID_U8 dst_id, int n) {
    cudaMemcpyToSymbol(DEV_LUT, LUT_majority, sizeof(unsigned char)*512);

    NppiSize target_thread = {32, 32};
    int block_width = tunits.getWidth() / target_thread.width;
    int block_height = tunits.getHeight() / target_thread.height;
    int block_size = block_width * block_height;
    dim3 blocks(block_width, block_height);
    dim3 threads(target_thread.width, target_thread.height);

    TexID_U8 tsid = src_id;
    TexID_U8 tmid1 = SBUFFER2_U8;
    cudaMemcpy(tunits.getBuffer(tmid1),
               tunits.getBuffer(src_id),
               tunits.u8Pitch()*tunits.getHeight(),
               cudaMemcpyDeviceToDevice);
    TexID_U8 tmid2 = SBUFFER3_U8;
    for(int i =0; i < n; i++) {
        applyLUT<<<blocks, threads>>>(
                tunits.getTexture(tmid1),
                tunits.getBuffer(tmid2), tunits.u8Pitch(), 0);
        TexID_U8 t = tmid1;
        tmid1 = tmid2;
        tmid2 = t;
    }

    cudaMemcpy(tunits.getBuffer(dst_id),
               tunits.getBuffer(tmid2),
               tunits.u8Pitch()*tunits.getHeight(),
               cudaMemcpyDeviceToDevice);
}

__host__
void prune(CurveDiscTextureUnits& tunits, TexID_U8 src_id, TexID_U8 dst_id, int n) {
    cudaMemcpyToSymbol(DEV_LUT, LUT_spur_matlab, sizeof(unsigned char)*512);
    cudaMemset(tunits.getBuffer(SBUFFER4_U8), 0, tunits.u8Pitch()*tunits.getHeight());
    cudaMemset(tunits.getBuffer(SBUFFER5_U8), 0, tunits.u8Pitch()*tunits.getHeight());
    cudaMemcpy(tunits.getBuffer(dst_id),
               tunits.getBuffer(src_id),
               tunits.u8Pitch()*tunits.getHeight(),
               cudaMemcpyDeviceToDevice);
//    cudaMemcpy(tunits.getBuffer(SBUFFER5_U8),
//            tunits.getBuffer(src_id),
//            tunits.u8Pitch()*tunits.getHeight(),
//            cudaMemcpyDeviceToDevice);

    NppiSize target_thread = {32, 32};
    int block_width = tunits.getWidth() / target_thread.width;
    int block_height = tunits.getHeight() / target_thread.height;
    int block_size = block_width * block_height;
    dim3 blocks(block_width, block_height);
    dim3 threads(target_thread.width, target_thread.height);


    for(int i = 0; i < n; i++) {
        // Apply spur lookup table
        applyLUT_inv<<<blocks, threads>>>(
                tunits.getTexture(dst_id),
                tunits.getBuffer(SBUFFER4_U8), tunits.u8Pitch());
        // Remove Spur from edge image
        xor_8u_step<<<blocks, threads>>>(tunits.getTexture(SBUFFER4_U8), tunits.getTexture(dst_id),
                                    tunits.getBuffer(dst_id), tunits.u8Pitch(), false, false);
//        tunits.outputTexture(dst_id);


        // In the second field, remove any of the original end points
        // that are still end points.
        applyLUT_inv<<<blocks, threads>>>(
                tunits.getTexture(dst_id),
                tunits.getBuffer(SBUFFER5_U8), tunits.u8Pitch());
        and_8u<<<blocks, threads>>>(tunits.getTexture(SBUFFER4_U8), tunits.getTexture(SBUFFER5_U8),
                                    tunits.getBuffer(SBUFFER3_U8), tunits.u8Pitch());
        xor_8u_step<<<blocks, threads>>>(tunits.getTexture(SBUFFER3_U8), tunits.getTexture(dst_id),
                                    tunits.getBuffer(dst_id), tunits.u8Pitch(), false, true);


        applyLUT_inv<<<blocks, threads>>>(
                tunits.getTexture(dst_id),
                tunits.getBuffer(SBUFFER5_U8), tunits.u8Pitch());
        and_8u<<<blocks, threads>>>(tunits.getTexture(SBUFFER4_U8), tunits.getTexture(SBUFFER5_U8),
                                    tunits.getBuffer(SBUFFER3_U8), tunits.u8Pitch());
        xor_8u_step<<<blocks, threads>>>(tunits.getTexture(SBUFFER3_U8), tunits.getTexture(dst_id),
                                         tunits.getBuffer(dst_id), tunits.u8Pitch(), true, false);

        applyLUT_inv<<<blocks, threads>>>(
                tunits.getTexture(dst_id),
                tunits.getBuffer(SBUFFER5_U8), tunits.u8Pitch());
        and_8u<<<blocks, threads>>>(tunits.getTexture(SBUFFER4_U8), tunits.getTexture(SBUFFER5_U8),
                                    tunits.getBuffer(SBUFFER3_U8), tunits.u8Pitch());
        xor_8u_step<<<blocks, threads>>>(tunits.getTexture(SBUFFER3_U8), tunits.getTexture(dst_id),
                                         tunits.getBuffer(dst_id), tunits.u8Pitch(), true, true);
    }

//    cv::waitKey(0);

}

__host__
void growEndpoints(CurveDiscTextureUnits& tunits, TexID_U8 edges, TexID_U8 dst_img){
    NppiSize target_thread = {32, 32};
    int block_width = tunits.getWidth() / target_thread.width;
    int block_height = tunits.getHeight() / target_thread.height;
    int block_size = block_width * block_height;
    dim3 blocks(block_width, block_height);
    dim3 threads(target_thread.width, target_thread.height);

    // Get Edge Range
//    float min, max;
//    determineTextureRange<<<blocks, threads>>>()

    // Convert edges to uchar
//    convert_32f_8u_scaled<<<blocks, threads>>>(
//            tunits.getTexture(edges),
//            tunits.getTexture(DEPTH_F32),
//            tunits.getBuffer(SBUFFER2_U8),
//            tunits.u8Pitch(),
//            0, 100
//            );
    tunits.zero(SBUFFER1_U8);
    tunits.zero(SBUFFER2_U8);

    prune(tunits, edges, SBUFFER1_U8, 3);
    tunits.outputTexture(SBUFFER1_U8, "de spurs");
    despeckle(tunits, SBUFFER1_U8, SBUFFER2_U8, 1);
    tunits.outputTexture(SBUFFER2_U8, "end points");

     gradientFollow2<<<blocks, threads>>>(
             tunits.getTexture(XGRAD_F32),
             tunits.getTexture(YGRAD_F32),
             tunits.getTexture(SBUFFER1_U8),
             tunits.getBuffer(dst_img),
             tunits.f32Pitch(),
             20
             );

}

__host__
void thin(CurveDiscTextureUnits& tunits, TexID_U8 src_id, TexID_U8 dst_id, int n){
    cudaMemcpyToSymbol(DEV_LUT, LUT_thin1, sizeof(unsigned char)*1024);

    NppiSize target_thread = {32, 32};
    int block_width = tunits.getWidth() / target_thread.width;
    int block_height = tunits.getHeight() / target_thread.height;
    int block_size = block_width * block_height;
    dim3 blocks(block_width, block_height);
    dim3 threads(target_thread.width, target_thread.height);

    applyLUT<<<blocks, threads>>>(
            tunits.getTexture(src_id),
            tunits.getBuffer(dst_id), tunits.u8Pitch(), 0);

    while(--n) {
        // Perform Thin
        applyLUT<<<blocks, threads>>>(
                tunits.getTexture(dst_id),
                tunits.getBuffer(SBUFFER2_U8), tunits.u8Pitch(), 512);

        applyLUT<<<blocks, threads>>>(
                tunits.getTexture(SBUFFER2_U8),
                tunits.getBuffer(dst_id), tunits.u8Pitch(), 0);
    }

}

__host__
void t_setupDepthMap(int width, int height, cv::Mat depth_map, cudaTextureObject_t& dev_depth_tex){
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
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&dev_depth_tex, &texRes, &texDescr, NULL));
}

void setupCudaTexture(int width, int height, void* cuda_img, size_t pitch, cudaTextureObject_t& dev_depth_tex,
        cudaChannelFormatDesc chnl_desc = cudaCreateChannelDesc<float>()){
    /// Intialize Resoure Descriptor
    cudaResourceDesc texRes;
    memset(&texRes, 0x0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr   = cuda_img;
    texRes.res.pitch2D.desc     = chnl_desc;
    texRes.res.pitch2D.width    = width;
    texRes.res.pitch2D.height   = height;
    texRes.res.pitch2D.pitchInBytes = pitch;

    /// Initialize Texture Descriptor
    cudaTextureDesc texDescr;
    memset(&texDescr, 0x0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&dev_depth_tex, &texRes, &texDescr, NULL));
}


__host__
cv::Mat launchCurveDiscOprKernel(cv::Mat& depth_map, CurveDiscTextureUnits& tex_units){

#ifdef DEBUG_CD_OPR_OUTPUT
    cv::Mat td_img;
    cv::normalize(depth_map, td_img, 0, 255, CV_MINMAX, CV_8U);
    cv::imshow("CD: Initial Image", td_img);
#endif

    NppStatus res;
    // Define image dimensions
    int height = depth_map.rows;
    int width = depth_map.cols;
    NppiSize src_size = {width, height};
    int pitch = tex_units.f32Pitch();
    int u8_pitch = tex_units.u8Pitch();

    // Cuda Kernel Thread dimensions
    NppiSize target_thread = {32, 32};
    int block_width = src_size.width/target_thread.width;
    int block_height = src_size.height / target_thread.height;
    int block_size = block_width * block_height;
    dim3 blocks(block_width, block_height);
    dim3 threads(target_thread.width, target_thread.height);


    int canny_buffer_size = 0;
    Npp8u* canny_buffer;

    nppiFilterCannyBorderGetBufferSize(src_size, &canny_buffer_size);
    cudaMalloc(&canny_buffer, canny_buffer_size);


    // Copy image to device Texture Target
    checkCudaErrors(cudaMemcpy2D(
            tex_units.getBuffer(SBUFFER1_F32), tex_units.f32Pitch(), depth_map.ptr(),
            depth_map.step, sizeof(float)*width, height, cudaMemcpyHostToDevice));

    // Fill Invalid Regions on depth image
    for(int i = 0; i < 10; i++){
        checkCudaErrors(cudaDeviceSynchronize());
        if(i%2 == 0){
            interpolate_invalid_regions<<<blocks, threads>>>(
                    tex_units.getTexture(SBUFFER1_F32),
                    tex_units.getBuffer(SBUFFER2_F32),
                    tex_units.f32Pitch());
        }else{
            interpolate_invalid_regions<<<blocks, threads>>>(
                    tex_units.getTexture(SBUFFER2_F32),
                    tex_units.getBuffer(SBUFFER1_F32),
                    tex_units.f32Pitch());
        }
    }

    // Average depth frame
    //tex_units.outputTexture(SBUFFER2_F32, "depth");
    averageDepthFrame(tex_units, SBUFFER2_F32, SBUFFER1_F32);
    //tex_units.outputTexture(SBUFFER1_F32, "averaged depth");
    //cv::waitKey(0);

    // Convert depth image to unsigned 8 to perform canny// Perform canny on depth image
    float depth_min, depth_max;
    determineTextureRange(tex_units.getTexture(SBUFFER1_F32), src_size, depth_max, depth_min);
    dim3 u8_threads = dim3(target_thread.width/4, target_thread.height); // each thread is responsible for 4 pixels

    checkCudaErrors(cudaDeviceSynchronize());
    convert_32f_8u_scaled<<<blocks, u8_threads>>>(
            tex_units.getTexture(SBUFFER1_F32),
            tex_units.getTexture(SBUFFER1_F32),
            tex_units.getBuffer(DEPTH_U8), tex_units.u8Pitch(), depth_min, depth_max);

    // Perform Canny for Depth Discontinuity Edges
    float dd_high_thresh = 0.07;
    float dd_low_thresh = dd_high_thresh * 0.9;
    nppiFilterCannyBorder_8u_C1R(
            tex_units.getBuffer(DEPTH_U8),
            tex_units.u8Pitch(),
            src_size,
            {0, 0},
            tex_units.getBuffer(DD_U8),
            tex_units.u8Pitch(),
            {width, height},
            NppiDifferentialKernel::NPP_FILTER_SOBEL,
            NppiMaskSize::NPP_MASK_SIZE_3_X_3,
            dd_low_thresh*255,
            dd_high_thresh*255,
            NppiNorm::nppiNormL2,
            NppiBorderType::NPP_BORDER_REPLICATE,
            canny_buffer
    );

    // Convert Depth Discontinuity Edges to F32
    convert_8u_32f_scaled<<<blocks, threads>>>(
            tex_units.getTexture(DD_U8),
            tex_units.getBuffer(DD_F32),
            tex_units.f32Pitch(), 0, 1.0/255.0);


    // Filter Input Image
    wiener_opr(tex_units, SBUFFER1_F32, DEPTH_F32, 5); // filter valid regions
    wiener_opr(tex_units, SBUFFER1_F32, DEPTH_F32, -1); // perform global region filtering

//    tex_units.outputTexture(DEPTH_F32, "Depth 2 Frame");
#ifdef DEBUG_CD_OPR_OUTPUT
    tex_units.outputTexture(DEPTH_F32, "fdepth");
    tex_units.outputTexture(DD_U8, "DD");
#endif

    // Perform Gradient
    sobel_shared<<<blocks, threads>>>(
            tex_units.getTexture(DEPTH_F32),
            tex_units.getTexture(DD_U8),
            tex_units.getBuffer(SBUFFER1_F32),
            tex_units.getBuffer(SBUFFER2_F32),
            tex_units.f32Pitch());
    // Filter X Gradient
    wiener_opr(tex_units, SBUFFER1_F32, XGRAD_F32, 5);
    // Filter Y Gradient ygrad
    wiener_opr(tex_units, SBUFFER2_F32, YGRAD_F32, 5);

    // Generate Gradient Normals
    gradient_normals<<<blocks, threads>>>(
            tex_units.getTexture(DEPTH_F32), src_size,
            tex_units.getTexture(XGRAD_F32), tex_units.getTexture(YGRAD_F32),
            tex_units.getBuffer(SBUFFER1_F32C4), tex_units.f32c4Pitch());
    wiener_opr(tex_units, SBUFFER1_F32C4, NORMAL_F32C4, 21);
    // Get contours minimums regions using normals (this is aching to regions with smallest magnitudes)
    normal_vector_seperation<<<blocks, threads>>>(
            tex_units.getTexture(NORMAL_F32C4), src_size,
            tex_units.getBuffer(HNORMEDGE_F32), tex_units.getBuffer(VNORMEDGE_F32), tex_units.f32Pitch());

    cudaMemset(tex_units.getBuffer(SBUFFER1_F32), 0, tex_units.f32Pitch()*height);
    convert_32f_8u_scaled<<<blocks, u8_threads>>>(
            tex_units.getTexture(HNORMEDGE_F32), tex_units.getTexture(DEPTH_F32),
            tex_units.getBuffer(SBUFFER1_U8), tex_units.u8Pitch(), 0, 100);

    convert_32f_8u_scaled<<<blocks, u8_threads>>>(
            tex_units.getTexture(VNORMEDGE_F32), tex_units.getTexture(DEPTH_F32),
            tex_units.getBuffer(SBUFFER2_U8), tex_units.u8Pitch(), 0, 100);

    or_8u<<<blocks, threads>>>(
            tex_units.getTexture(SBUFFER1_U8),
            tex_units.getTexture(SBUFFER2_U8),
            tex_units.getBuffer(NORMEDGE_U8), tex_units.u8Pitch());
    //thin(tex_units, SBUFFER3_U8, NORMEDGE_U8, 10);
//    tex_units.outputTexture(NORMEDGE_U8, "norm t");
    checkCudaErrors(cudaDeviceSynchronize());

    // Perform Gradient Mirroring (lrgrad -> sbuffer1 | udgrad -> sbuffer2)
    gradientOperation<<<height, 16>>>(
            tex_units.getBuffer(XGRAD_F32),
            tex_units.getBuffer(YGRAD_F32), src_size,
            tex_units.getBuffer(SBUFFER1_F32),
            tex_units.getBuffer(SBUFFER2_F32), pitch, 0.8);

    // Perform Filtering (lrgrad -> sbuffer5 | udgrad -> sbuffer6)
    wiener_opr(tex_units, SBUFFER1_F32, LRDIR_F32, 3);
    wiener_opr(tex_units, SBUFFER2_F32, UDDIR_F32, 3);


#ifdef DEBUG_CD_OPR_OUTPUT
    cv::FileStorage storage;
    tex_units.outputTexture(XGRAD_F32, "xgrad");
    tex_units.outputTexture(YGRAD_F32, "ygrad");
//    tex_units.outputTexture(UDDIR_F32, "uddir");
//    tex_units.outputTexture(LRDIR_F32, "lrdir");
    tex_units.outputTexture(SBUFFER1_F32C4, "rnormals");
    tex_units.outputTexture(NORMAL_F32C4, "normals");
    tex_units.outputTexture(NORMEDGE_U8, "normedge");
    tex_units.outputTexture(HNORMEDGE_F32, "hnormedge");
    tex_units.outputTexture(VNORMEDGE_F32, "vnormedge");
#endif
    float xrange_max, xrange_min;
    determineTextureRange(tex_units.getTexture(LRDIR_F32), src_size, xrange_max, xrange_min);
    float yrange_max, yrange_min;
    determineTextureRange(tex_units.getTexture(UDDIR_F32), src_size, yrange_max, yrange_min);

    float highThresh = 0.4;
    float lowThresh = 0.98 * highThresh;


    convert_32f_8u_scaled<<<blocks, u8_threads>>>(
            tex_units.getTexture(LRDIR_F32), tex_units.getTexture(DEPTH_F32),
            tex_units.getBuffer(SBUFFER1_U8), tex_units.u8Pitch(), xrange_min, xrange_max);
    checkCudaErrors(cudaDeviceSynchronize());
    // nppiConvert_32f8u_C1R(sbuffer1, pitch, u8_x_buffers[0], u8_pitch, src_size, NppRoundMode::NPP_RND_NEAR);
    nppiFilterCannyBorder_8u_C1R(
            tex_units.getBuffer(SBUFFER1_U8),
            tex_units.u8Pitch(),
            src_size,
            {0, 0},
            tex_units.getBuffer(LREDGE_U8),
            tex_units.u8Pitch(),
            {width, height},
            NppiDifferentialKernel::NPP_FILTER_SOBEL,
            NppiMaskSize::NPP_MASK_SIZE_3_X_3,
            lowThresh* 255,
            highThresh * 255,
            NppiNorm::nppiNormL1,
            NppiBorderType::NPP_BORDER_REPLICATE,
            canny_buffer
            );

    convert_32f_8u_scaled<<<blocks, u8_threads>>>(
            tex_units.getTexture(UDDIR_F32), tex_units.getTexture(DEPTH_F32),
            tex_units.getBuffer(SBUFFER2_U8), tex_units.u8Pitch(), yrange_min, yrange_max);
    // nppiConvert_32f8u_C1R(sbuffer1, pitch, u8_x_buffers[0], u8_pitch, src_size, NppRoundMode::NPP_RND_NEAR);
    nppiFilterCannyBorder_8u_C1R(
            tex_units.getBuffer(SBUFFER2_U8),
            tex_units.u8Pitch(),
            src_size,
            {0, 0},
            tex_units.getBuffer(UDEDGE_U8),
            tex_units.u8Pitch(),
            {width, height},
            NppiDifferentialKernel::NPP_FILTER_SOBEL,
            NppiMaskSize::NPP_MASK_SIZE_3_X_3,
            lowThresh * 255,
            highThresh * 255,
            NppiNorm::nppiNormL2,
            NppiBorderType::NPP_BORDER_REPLICATE,
            canny_buffer
    );

//    tex_units.outputTexture(LREDGE_U8, "LR EDGE");
//    tex_units.outputTexture(UDEDGE_U8, "UD EDGE");

    or_8u<<<blocks, threads>>>(
            tex_units.getTexture(LREDGE_U8),
            tex_units.getTexture(UDEDGE_U8),
            tex_units.getBuffer(SBUFFER7_U8),
            tex_units.u8Pitch());
    despeckle(tex_units, SBUFFER7_U8, SBUFFER7_U8, 2);
    fatten(tex_units, SBUFFER7_U8, SBUFFER7_U8, 1);

    prune(tex_units, NORMEDGE_U8, SBUFFER9_U8, 3);
    thin(tex_units, SBUFFER9_U8, SBUFFER9_U8, 2);
//    tex_units.outputTexture(SBUFFER9_U8, "edges1");

    or_8u<<<blocks, threads>>>(tex_units.getTexture(SBUFFER7_U8),
            tex_units.getTexture(SBUFFER9_U8),
            tex_units.getBuffer(SBUFFER8_U8), tex_units.u8Pitch());
//    tex_units.outputTexture(SBUFFER8_U8, "edgesf");

    majority(tex_units, SBUFFER8_U8, SBUFFER9_U8, 10);
//    tex_units.outputTexture(SBUFFER8_U8, "majority");

    thin(tex_units, SBUFFER9_U8, SBUFFER8_U8, 10);
//    tex_units.outputTexture(SBUFFER8_U8, "thinned");
    prune(tex_units,SBUFFER8_U8, SBUFFER7_U8, 12);
    despeckle(tex_units, SBUFFER7_U8, SBUFFER8_U8, 1);
//    tex_units.outputTexture(SBUFFER8_U8, "spurred");

//    cv::imwrite("./results/spur.png", tex_units.toCVMat(SBUFFER7_U8));
//    cv::waitKey(0);

#ifdef DEBUG_CD_OPR_OUTPUT
    tex_units.outputTexture(LREDGE_U8, "lredge");
    tex_units.outputTexture(UDEDGE_U8, "udedge");
    cv::waitKey(0);
#endif

    cv::Mat tres = tex_units.toCVMat(SBUFFER8_U8);
    cv::imshow("results", tres);
    //cv::waitKey(0);
    return tres;

    /**
     * Perform Closing, ske, and despuring to clean contours
     */

//
//    cv::Mat hori_edge(height, width, CV_8U);
//    cv::Canny(stage3_lrgdir_img, hori_edge, 255*lowThresh, 255*highThresh);
//
//    cv::Mat vert_edge(height, width, CV_8U);
//    cv::Canny(stage3_udgdir_img, vert_edge, 255*lowThresh, 255*highThresh);

//    cv::Mat result = hori_edge | vert_edge;

#ifdef DEBUG_CD_OPR_OUTPUT
//    cv::imshow("Stage 4 CD Hori Results", hori_edge);
//    cv::imshow("Stage 4 CD Vert Results", vert_edge);
//    cv::imwrite("cd_res.png", result);
//    cv::imshow("Final CD Results", result);
//    cv::waitKey(0);
#endif
}


void bwlookup( const cv::Mat & in, cv::Mat & out, const cv::Mat & lut, int bordertype=cv::BORDER_CONSTANT, cv::Scalar px = cv::Scalar(0) )
{
    if ( in.type() != CV_8UC1 )
        CV_Error(CV_StsError, "er");
    if ( lut.type() != CV_8UC1 || lut.rows*lut.cols!=512 || !lut.isContinuous() )
        CV_Error(CV_StsError, "lut size != 512" );
    if ( out.type() != in.type() || out.size() != in.size() )
        out = cv::Mat( in.size(), in.type() );

    const unsigned char * _lut = lut.data;
    cv::Mat t;
    cv::copyMakeBorder( in,t,1,1,1,1,bordertype,px);
    const int rows=in.rows+1;
    const int cols=in.cols+1;
    for ( int y=1;y<rows;++y)
    {
        for ( int x=1;x<cols;++x)
        {
            int L = 0;
            const int jmax=y+1;
#if 1 // row-major order
            for ( int j=y-1, k=1; j<=jmax; ++j, k<<=3 )
			{
				const unsigned char * p = t.ptr<unsigned char>(j) + x-1;
				for ( unsigned int u=0;u<3;++u )
				{
					if ( p[u] )
						L += (k<<u);
                }
            }
#else // column-major order (MATLAB)
            for ( int j=y-1, k=1; j<=jmax; ++j, k<<=1 )
            {
                const unsigned char * p = t.ptr<unsigned char>(j) + x-1;
                for ( unsigned int u=0;u<3;++u )
                {
                    if ( p[u] )
                        L += (k<<3*u);

                }
            }
#endif
            out.at<unsigned char>(y-1,x-1)=_lut[ L ];
        }
    }
}

cv::Mat skeletonize(cv::Mat& in, int n){
    cv::Mat scratch1 = in / 255;
    cv::Mat morph_out;
    if (n <= 0)
        CV_Error(CV_StsError, "er");

    while(--n) {
        cv::Mat x = (scratch1 != 0) / 255;
        // Perform Morphological skeletonization
        cv::Mat lut(1, 512, CV_8UC1, (void *) LUT_skel1);
        bwlookup(scratch1, morph_out, lut);

        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_skel2);
        bwlookup(morph_out, scratch1, lut);
        morph_out = x & scratch1;

        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_bridge1);
        bwlookup(morph_out, scratch1, lut);
    }

    return scratch1;
}

cv::Mat shrink(cv::Mat& in, int n){
    cv::Mat scratch1 = in.clone();
    cv::Mat morph_out;
    cv::Mat lut;
    if (n <= 0)
        CV_Error(CV_StsError, "er");

    while(--n) {
        // Perform Shrink
        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_shrink1);
        bwlookup(scratch1, morph_out, lut);
        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_shrink2);
        bwlookup(morph_out, scratch1, lut);
        scratch1 &= in;
    }

    return scratch1;
}

cv::Mat clean(cv::Mat& in){
    const int CONECTIVITY = 8;
    cv::Mat morph;
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    kernel.at<unsigned char>(1, 1) = CONECTIVITY;
    cv::filter2D(in, morph, -1, kernel);

    return morph > CONECTIVITY;
}

cv::Mat hbreak(cv::Mat& in, int n){
    cv::Mat scratch1 = in.clone();
    cv::Mat morph_out;
    cv::Mat lut;

    if (n <= 0)
        CV_Error(CV_StsError, "er");

    while(--n) {
        // Perform hbreak
        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_hbreak1);
        bwlookup(scratch1, morph_out, lut);
        scratch1 = morph_out.clone();
    }
    return scratch1;
}

cv::Mat spur(cv::Mat& in, int n){
    cv::Mat scratch1 = in.clone();
    cv::Mat morph_out;
    cv::Mat lut;

    if (n <= 0)
        CV_Error(CV_StsError, "er");

    while(--n) {
        // Perform hbreak
        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_spur1);
        bwlookup(scratch1, morph_out, lut);
        scratch1 = morph_out.clone();
    }
    return scratch1;
}

cv::Mat thin(cv::Mat& in, int n){
    cv::Mat scratch1 = in.clone();
    cv::Mat morph_out;
    cv::Mat lut;
    if (n <= 0)
        CV_Error(CV_StsError, "er");

    while(--n) {
        // Perform Shrink
        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_thin1);
        bwlookup(scratch1, morph_out, lut);
        lut = cv::Mat(1, 512, CV_8UC1, (void *) LUT_thin2);
        bwlookup(morph_out, scratch1, lut);
    }

    return scratch1;
}

extern "C"
cv::Mat cleanDiscontinuityOpr(cv::Mat& disc_img){
    cv::Mat x = disc_img;
    cv::morphologyEx(x, x, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    cv::Mat img = skeletonize(x, 10);
//    img = shrink(img, 10);
//    cv::imshow("test1", img*255);
//    img = clean(img);
//    img = hbreak(img, 20);
//    cv::imshow("test2", img*255);
    cv::morphologyEx(img, img, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
//    img = clean(img);
    img = thin(img, 10);
    img = spur(img, 10);

    return img;
}

extern "C"
cv::Mat cuCurveDiscOperation(cv::Mat& depth_map){
    static CurveDiscTextureUnits* tex_units;
    if(tex_units == nullptr){
        tex_units = new CurveDiscTextureUnits(depth_map.cols, depth_map.rows);
    }

    cv::Mat x = launchCurveDiscOprKernel(depth_map, *tex_units);

    tex_units->zeroElements();


#ifdef DEBUG_CD_OPR_OUTPUT
    cv::imshow("Final CD", x);
    cv::imwrite("./results/final_cd.png", x);
    cv::waitKey(0);
#endif

    return x;
}