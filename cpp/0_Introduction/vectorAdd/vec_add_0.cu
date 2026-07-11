#include <cuda_runtime.h>
#include <cmath>
#include <random>

#define CUDA_CHECK(call) \
    do{                     \
        cudaError_t err=call; \
        if(err!=cudaSuccess){  \
            printf("cuda error:%s\n",cudaGetErrorString(err)); \
            return; \
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
    int N=5*1e4;
    int size=N*sizeof(float);
    float *h_a,*h_b,*h_c,*h_result;
    h_a=(float*)malloc(size);
    h_b=(float*)malloc(size);
    h_c=(float*)malloc(size);
    h_result=(float*)malloc(size);

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
    int grid_size=(block_size+N-1)/block_size;
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
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d_ms=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms,start,stop));
    double h2d_bytes=2*size*repeat;
    double h2d_gbs=(h2d_bytes/(h2d_ms/1000))/1e9;
    printf("h2d avg time:%.5f ms,h2d bandwith :%.2f GB/s\n",h2d_ms/repeat,h2d_gbs);


    // kernel bandwith
    CUDA_CHECK(cudaEventRecord(start));
    for(int i=0;i<repeat;i++){
        device_vec_add<<<grid_size,block_size>>>(d_a,d_b,d_c,N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float kernel_ms=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms,start,stop));
    double kernel_bytes=3*size*repeat;
    double kernel_gbs=(kernel_bytes/(kernel_ms/1000))/1e9;
    printf("kernel avg time:%.5f ms,kernel bandwith:%.2f GB/s\n",kernel_ms/1000,kernel_gbs);

    // D2H bandwith
    CUDA_CHECK(cudaEventRecord(start));
    for(int i=0;i<repeat;i++){
        CUDA_CHECK(cudaMemcpy(h_result,d_c,size,cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h_ms=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms,start,stop));
    double d2h_bytes=size*repeat;
    double d2h_gbs=(d2h_bytes/(d2h_ms/1000))/1e9;
    printf("d2h avg time:%.5f ms,d2h bandwith:%.2f GB/s\n",d2h_ms/1000,d2h_gbs);

    // cmp
    bool flag=true;
    for(int i=0;i<N;i++){
        if(std::fabs(h_result[i]-h_c[i])>1e-6){
            flag=false;
            printf("data idx:%d, host item:%.6f,device item:%.6f\n",i,h_c[i],h_result[i]);
            break;
        }
    }

    if(!flag){
        printf("data not match!\n");
    }else{
        printf("data match!\n");
    }

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_result);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}