#include <cudaruntime.h>
#include <stdio.h>


#define CUDA_CHECK(call)    \
    do {                     \
        cudaError_t err=call; \
        if(err!=cudaSuccess){  \
            printf("cuda error:%s\n",cudaGetErrorString(err));   \
        }                                                         \
    }while(0);


float elapsedTime(cudaEvent_t start,cudaEvent_t end){
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms,start,end));
    return ms;
}


