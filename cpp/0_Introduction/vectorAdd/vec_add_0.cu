#include <cuda_runtime.h>
#include <cmath>
#include <random>

#define CUDA_CHECK(call) \
    do{                     \
        cudaError_t err=call; \
        if(err!=cudaSuccess){  \
            printf("cuda error:%s\n",cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE); \
        } \
    }while(0); 


void init_host_data(float *h_a,float *h_b,float *h_c,int N){
    std::mt19937 engine(42);
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    for(int i=0;i<N;i++){
        h_a[i]=dist(engine);
        h_b[i]=dist(engine);
        h_c[i]=h_a[i]+h_b[i];
    }
}

void __global__ device_vec_add(float *a,float *b,float *c,int N){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<N){
        c[idx]=a[idx]+b[idx];
    }
}

int main(){
    constexpr int N = 5'000'000'0;
    size_t size=static_cast<size_t>(N)*sizeof(float);
    float *h_a,*h_b,*h_c,*h_result;
    // h_a=(float*)malloc(size);
    // h_b=(float*)malloc(size);
    // h_c=(float*)malloc(size);
    // h_result=(float*)malloc(size);
    CUDA_CHECK(cudaMallocHost(&h_a,size));
    CUDA_CHECK(cudaMallocHost(&h_b,size));
    CUDA_CHECK(cudaMallocHost(&h_c,size));
    CUDA_CHECK(cudaMallocHost(&h_result,size));

    init_host_data(h_a,h_b,h_c,N);

    float *d_a,*d_b,*d_c;
    CUDA_CHECK(cudaMalloc(&d_a,size))
    CUDA_CHECK(cudaMalloc(&d_b,size))
    CUDA_CHECK(cudaMalloc(&d_c,size))

    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warm up
    CUDA_CHECK(cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice));
    int block_size=256;
    int grid_size=(N+block_size-1)/block_size;
    device_vec_add<<<grid_size,block_size>>>(d_a,d_b,d_c,N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // H2D
    int repeat=100;
    CUDA_CHECK(cudaEventRecord(start));
    for(int i=0;i<repeat;i++){
        CUDA_CHECK(cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d_ms=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms,start,stop));
    double h2d_bytes=2.0f*static_cast<double>(size)*repeat;
    double h2d_gbs=(h2d_bytes/(h2d_ms/1000))/1e9;
    printf("h2d avg time:%.5f ms,h2d bandwith:%.2f GB/s\n",h2d_ms/repeat,h2d_gbs);


    // kernel bandwith
    CUDA_CHECK(cudaEventRecord(start));
    for(int i=0;i<repeat;i++){
        device_vec_add<<<grid_size,block_size>>>(d_a,d_b,d_c,N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float kernel_ms=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms,start,stop));
    double kernel_bytes=3.0f*static_cast<double>(size)*repeat;
    double kernel_gbs=(kernel_bytes/(kernel_ms/1000))/1e9;
    printf("kernel avg time:%.5f ms,kernel bandwith:%.2f GB/s\n",kernel_ms/repeat,kernel_gbs);

    // D2H bandwith
    CUDA_CHECK(cudaEventRecord(start));
    for(int i=0;i<repeat;i++){
        CUDA_CHECK(cudaMemcpy(h_result,d_c,size,cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h_ms=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms,start,stop));
    double d2h_bytes=static_cast<double>(size)*repeat;
    double d2h_gbs=(d2h_bytes/(d2h_ms/1000))/1e9;
    printf("d2h avg time:%.5f ms,d2h bandwith:%.2f GB/s\n",d2h_ms/repeat,d2h_gbs);

    // cmp
    bool flag=true;
    float max_offset=0.0f;
    for(int i=0;i<N;i++){
        float offset=std::fabs(h_result[i]-h_c[i]);
        max_offset=std::max(offset,max_offset);
        if(offset>1e-6){
            flag=false;
            printf("data idx:%d, host item:%.6f,device item:%.6f\n",i,h_c[i],h_result[i]);
        }
    }
    printf("max offset:%.5f\n",max_offset);

    if(!flag){
        printf("data not match!\n");
    }else{
        printf("data match!\n");
    }

    // free(h_a);
    // free(h_b);
    // free(h_c);
    // free(h_result);
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    CUDA_CHECK(cudaFreeHost(h_result));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}