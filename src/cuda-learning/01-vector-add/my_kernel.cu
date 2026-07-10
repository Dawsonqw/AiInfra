#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>


void init_data(float* val,int size){
    for(int i=0;i<size;i++){
        val[i]=2.0f*i;
    }
}

bool cmp_data(float* left,float *right,int num){
    for(int i=0;i<num;i++){
        if(std::fabs(left[i]-right[i])>1e-6){
            return false;
        }
    }
    return true;
}


void __global__ device_vec_add(float *left,float *right,float *result,int size){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<size){
        result[idx]=left[idx]+right[idx];
    }
}

void host_add(float *left,float *right,float *result,int size){
    for(int i=0;i<size;i++){
        result[i]=left[i]+right[i];
    }
}

#define CHECK_CUDA(call) \
    do{                    \
       cudaError_t err=call; \
       if(err!=cudaSuccess){ \
        printf("cuda error:%s\n",cudaGetErrorString(err)); \
        exit(1); \
       } \
    }while(0);


float elapsed_time(cudaEvent_t start,cudaEvent_t end){
    float ms=0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms,start,end));
    return ms;
}

int main(){
    int element_num = 1<<28;
    size_t element_size=sizeof(float)*element_num;

    printf("element_num: %d\n",element_num);
    printf("element bytes: %d Mb\n",element_size/(1024*1024));

    float *host_add_left=(float*)malloc(element_size);
    float *host_add_right=(float*)malloc(element_size);
    float *host_result=(float*)malloc(element_size);
    float *cmp_result=(float*)malloc(element_size);

    init_data(host_add_left,element_num);
    init_data(host_add_right,element_num);
    host_add(host_add_left,host_add_right,host_result,element_num);

    float *device_add_left,*device_add_right,*device_add_result;
    CHECK_CUDA(cudaMalloc(&device_add_left,element_size));
    CHECK_CUDA(cudaMalloc(&device_add_right,element_size));
    CHECK_CUDA(cudaMalloc(&device_add_result,element_size));


    int threads=256;
    int blocks=(element_num+threads-1)/threads;

    cudaEvent_t start ,end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    // warm up
    CHECK_CUDA(cudaMemcpy(device_add_left,host_add_left,element_size,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_add_right,host_add_right,element_size,cudaMemcpyHostToDevice));
    device_vec_add<<<blocks,threads>>>(device_add_left,device_add_right,device_add_result,element_num);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int repeat=100;

    // H2D bandwith
    CHECK_CUDA(cudaEventRecord(start));

    for(int i=0;i<repeat;i++){
        CHECK_CUDA(cudaMemcpy(device_add_left,host_add_left,element_num,cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(device_add_right,host_add_right,element_num,cudaMemcpyHostToDevice));
    }

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));

    float h2d_ms=elapsed_time(start,end);
    double h2d_bytes=2.0*element_size*repeat;
    double h2d_gbps=h2d_bytes/(h2d_ms/1000)/1e9;

    printf("h2d totol time :%.3f ms\n",h2d_ms);
    printf("h2d bandwith :%.2f GB/s\n",h2d_gbps);


    // kernel bandwith

    CHECK_CUDA(cudaEventRecord(start));

    for(int i=0;i<repeat;i++){
        device_vec_add<<<blocks,threads>>>(device_add_left,device_add_right,device_add_result,element_num);
    }

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));
    CHECK_CUDA(cudaGetLastError());

    float kernel_ms=elapsed_time(start,end);
    double kernel_bytes=3.0f*element_size*repeat;
    double kernel_gbps = kernel_bytes / (kernel_ms / 1000.0) / 1e9;

    printf("Kernel time total: %.3f ms\n", kernel_ms);
    printf("Kernel time avg:   %.6f ms\n", kernel_ms / repeat);
    printf("Kernel bandwidth:  %.2f GB/s\n", kernel_gbps);

    // -------------------------
    // D2H bandwidth
    // -------------------------
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++) {
        CHECK_CUDA(cudaMemcpy(cmp_result, device_add_result, element_size, cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));

    float d2h_ms = elapsed_time(start, end);
    double d2h_bytes = 1.0 * element_size * repeat;
    double d2h_gbps = d2h_bytes / (d2h_ms / 1000.0) / 1e9;

    printf("D2H time total: %.3f ms\n", d2h_ms);
    printf("D2H bandwidth:  %.2f GB/s\n", d2h_gbps);

    // -------------------------
    // End-to-end: H2D + kernel + D2H
    // -------------------------
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++) {
        CHECK_CUDA(cudaMemcpy(device_add_left, host_add_left, element_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(device_add_right, host_add_right, element_size, cudaMemcpyHostToDevice));

        device_vec_add<<<blocks, threads>>>(device_add_left, device_add_right, device_add_result, element_num);

        CHECK_CUDA(cudaMemcpy(cmp_result, device_add_result, element_size, cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));
    CHECK_CUDA(cudaGetLastError());

    float e2e_ms = elapsed_time(start, end);
    double e2e_bytes = 3.0 * element_size * repeat;
    double e2e_gbps = e2e_bytes / (e2e_ms / 1000.0) / 1e9;

    printf("End-to-end time total: %.3f ms\n", e2e_ms);
    printf("End-to-end avg time:   %.6f ms\n", e2e_ms / repeat);
    printf("End-to-end bandwidth:  %.2f GB/s\n", e2e_gbps);

    if (cmp_data(host_result, cmp_result, element_num)) {
        printf("data match\n");
    } else {
        printf("no match\n");
    }

    CHECK_CUDA(cudaFree(device_add_left));
    CHECK_CUDA(cudaFree(device_add_right));
    CHECK_CUDA(cudaFree(device_add_result));

    CHECK_CUDA(cudaFreeHost(host_add_left));
    CHECK_CUDA(cudaFreeHost(host_add_right));
    CHECK_CUDA(cudaFreeHost(host_result));
    CHECK_CUDA(cudaFreeHost(cmp_result));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));


    return 0;
}