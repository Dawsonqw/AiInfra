#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void initialize_data(float *a,float *b,int N){
    for(int i=0;i<N;i++){
        a[i]=i;
        b[i]=i;
    }
}

void check_result(float *ref_cpu,float *ref_gpu,int N){
    double val=1e-8;
    bool flag=true;
    for(int i=0;i<N;i++){
        if(fabs(ref_cpu[i]-ref_gpu[i])>val){
            flag=false;
            break;
        }
    }
    if(flag){
        printf("match!\n");
    }else{
        printf("not match!\n");
    }

}


__global__ void vector_add(float *a,float *b,float *c,int N){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;

    if(idx<N){
        c[idx]=a[idx]+b[idx];
    }
}

void vector_add_cpu(float *a,float *b,float *c,int N){
    for(int i=0;i<N;i++){
        c[i]=a[i]+b[i];
    }
}

int main(){
    int N=256;


    float *h_a,*h_b,*h_c,*h_ref;
    size_t nBytes=N*sizeof(float);

    h_a=(float*)malloc(nBytes);
    h_b=(float*)malloc(nBytes);
    h_c=(float*)malloc(nBytes);
    h_ref=(float*)malloc(nBytes);

    initialize_data(h_a,h_b,N);

    float *d_a,*d_b,*d_c;
    cudaMalloc(&d_a,nBytes);
    cudaMalloc(&d_b,nBytes);
    cudaMalloc(&d_c,nBytes);

    cudaMemcpy(d_a,h_a,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,nBytes,cudaMemcpyHostToDevice);


    int threads=256;
    int blocks=(N+threads-1)/threads;


    vector_add<<<blocks,threads>>>(d_a,d_b,d_c,N);

    cudaMemcpy(h_ref,d_c,nBytes,cudaMemcpyDeviceToHost);

    vectro_add_cpu(h_a,h_b,h_c,N);
    check_result(h_c,h_ref,N);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_ref);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}