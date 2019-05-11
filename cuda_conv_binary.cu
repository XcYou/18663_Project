//
//  Alexandre Maros - 2016
//
//  Cuda Matrix Multiplication with Shared Memory.
//
//  nvcc cuda_matrix_shared.cu -o cs.o
//
//  Implemented by Alexandre Maros for learning purposes.
//  A version of this code using Global Memory is in here:
//  https://github.com/alepmaros/cuda_matrix_multiplication
//
//  Distributed under the MIT Lincese.


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <bitset>
#include <time.h>

#define NTHREADS_X 32
#define NTHREADS_Y 32
//#define N 3
//#define M 1
static const int input_size = 1000;
static const int DEPTH = 20;
static const int stride = 1;
static const int padding = 0;
static const int filter_size = 3;
static const int output_size = (input_size - filter_size + 2*padding)/stride + 1;
static const int comp_col = filter_size*DEPTH/32 + 1;
//static const int depth_output_size = (DEPTH - DEPTH + 2*padding)/stride + 1;
//#define THREADS_PER_BLOCK NTHREADS_X * NTHREADS_Y

//A macro used for error checking in CUDA function calls
//Credit to: http://stackoverflow.com/a/14038590 for the gpuErrchk macro.

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
__device__ int result_device[output_size];

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
//        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

__device__ void Mat_ACC(unsigned int C[filter_size][comp_col], int filter_size, int column){
    unsigned int counter = 0;
    unsigned int pop = filter_size*DEPTH;
    unsigned int threshold = filter_size*filter_size*DEPTH/2;
    __syncthreads();
    for (int i = 0; i<filter_size; i++) {
        for (int j = 0; j<comp_col; j++) {
            if(pop<32){
                counter += __popc(C[i][j]);
                counter = counter - (32 - pop);
            }
            else {
                counter += __popc(C[i][j]);
                pop = pop - 32;
            }
        }
    }
    if (counter > threshold){
        result_device[column] = 1;
    }
    else{
        result_device[column] = 0;
    }
    __syncthreads();
}

__device__ void Mat_XNOR(unsigned int A[filter_size][comp_col], unsigned int B[filter_size][comp_col], unsigned int C[filter_size][comp_col], int column){
    __syncthreads();
    for (unsigned int i=0; i<filter_size; i++){
        for(unsigned int j=0; j<comp_col; j++){
            C[i][j] = ~((A[i][j] ^ B[i][j]));
        }
    }
    Mat_ACC(C, filter_size, column);
    __syncthreads();
}

__global__ void convolve(unsigned int i_fmap[][input_size*DEPTH],
                         unsigned int filter[][filter_size*DEPTH],
                         unsigned int output[][output_size],
                         int padding, int stride){
//    printf("entering conv function\n");

    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int line =  blockIdx.y * blockDim.y + threadIdx.y;
//    printf("column = %d\n",column);
//    printf("line = %d\n",line);

    unsigned int pA_size, pB_size, pC_size;
//    unsigned int xA_size, xB_size, xC_size;
    unsigned int (*pA)[filter_size*DEPTH], (*pB)[filter_size*DEPTH], (*pC)[filter_size*DEPTH];
//    unsigned int** pA = new (unsigned int*)[filter_size];
//    unsigned int** pB = new (unsigned int*)[filter_size];
//    unsigned int** pC = new (unsigned int*)[filter_size];
//    for (int i=0; i<filter_size; i++){
//        pA[i] = new (unsigned int*)[filter_size];
//        pB[i] = new (unsigned int*)[filter_size];
//        pC[i] = new (unsigned int*)[filter_size];
//    }
    pA_size = filter_size*filter_size*DEPTH* sizeof(unsigned int);
    pB_size = filter_size*filter_size*DEPTH* sizeof(unsigned int);
    pC_size = filter_size*filter_size*DEPTH* sizeof(unsigned int);
//    xA_size = filter_size*comp_col * sizeof(unsigned int);
//    xB_size = filter_size*comp_col * sizeof(unsigned int);
//    xC_size = filter_size*comp_col * sizeof(unsigned int);
//    pA = (unsigned int **)malloc(pA_size);
//    pB = (unsigned int **)malloc(pB_size);
//    pC = (unsigned int **)malloc(pC_size);
//    printf("Before CUDAMALLOC function\n");
    cudaMalloc((void **) &pA, pA_size);
    cudaMalloc((void **) &pB, pB_size);
    cudaMalloc((void **) &pC, pC_size);
//    cudaMalloc((void **) &xA, xA_size);
//    cudaMalloc((void **) &xB, xB_size);
//    cudaMalloc((void **) &xC, xC_size);
//    printf("After CUDAMALLOC function\n");
//    printf("%u\n", filter[1][1]);
//    for (int fout_r = 0; fout_r<output_size; fout_r++) {
        for (int fout_c = 0; fout_c < output_size; fout_c++) {
            for (int fr = 0; fr < filter_size; fr++) {
                for (int fc = 0; fc < filter_size * DEPTH; fc++) {
                    //take values from input matrix
                    pB[fr][fc] = filter[fr][fc];
//            printf("for loop\n");

                }
            }
            for (int d = 0; d<DEPTH; d++) {
                for (int r = 0; r < filter_size; r++) {
                    for (int c = 0; c < filter_size; c++) {
                        pA[r][d*filter_size + c] = i_fmap[column * stride + r][d * input_size + fout_c*stride + c];
//                        printf("%u ", pA[r][d*filter_size + c]);
                    }
//                    printf("\n");
                }
//                printf("\n");
            }
//            memset(xA, 0, xA_size);
//            memset(xB, 0, xB_size);
//            memset(xC, 0, xC_size);
            Mat_XNOR(xA, xB, xC, column);
//    cudaThreadSynchronize();
//            printf("After XNOR function\n");
//            __syncthreads();
            output[column][fout_c] = result_device[column];
//            __syncthreads();
        }
//    }
//    pB = filter[0];
//    printf("Before XNOR function\n");
//    printf("result_conv=%d\n", output[0][0*DEPTH + 0]);

    cudaFree(pA);
    cudaFree(pB);
    cudaFree(pC);
}

int main(){

//    printf("pA = %d", A[0][0][0]);
//    dim3 numBlocks(1,1,1);
//    dim3 threadsPerBlock(N,N,DEPTH);
//cudaDeviceSynchronize();
//    MatAdd<<<numBlocks,threadsPerBlock>>>(pA,pB,pC);
//    cudaDeviceSynchronize();

//    unsigned int *a, *b, *c;
//    unsigned int *a_f, *b_f, *c_f;
    srand(time(0));

    unsigned int A[input_size][input_size*DEPTH];
    unsigned int B[filter_size][filter_size*DEPTH];
    unsigned int C[output_size][output_size];
    unsigned int xA[filter_size][comp_col], xB[filter_size][comp_col], xC[filter_size][comp_col];

    int a_nlines, a_ncolumns;
    int b_nlines, b_ncolumns;
//    int c_nlines, c_ncolumns;

    cudaEvent_t start, stop;
    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );
    //
    scanf("%d", &a_nlines);
    scanf("%d", &a_ncolumns);
    scanf("%d", &b_nlines);
    scanf("%d", &b_ncolumns);
    //
    unsigned int (*tA)[input_size*DEPTH], (*tB)[filter_size*DEPTH], (*tC)[output_size];

    gpuErrchk( cudaMalloc((void**)&tA, (input_size*input_size * DEPTH)*sizeof(unsigned int)));
    gpuErrchk( cudaMalloc((void**)&tB, (filter_size*filter_size * DEPTH)*sizeof(unsigned int)));
    gpuErrchk( cudaMalloc((void**)&tC, (output_size*output_size)*sizeof(unsigned int)));

    memset(C, 0, output_size*output_size*sizeof(unsigned int));
    //generate random input
    for (int layer = 0; layer < DEPTH; layer++){
        for (int row = 0; row < input_size; row++){
            for (int col = 0; col < input_size; col++){
//                scanf("%u", &a[i * a_ncolumns + j]);
                A[row][col+layer*input_size] = rand() % 2;
            }
        }
    }

    for (int layer = 0; layer < DEPTH; layer++){
        for (int row = 0; row < filter_size; row++){
            for (int col = 0; col < filter_size; col++){
                B[row][col+layer*filter_size] = rand() % 2;
            }
        }
    }
    printf("Input A = \n");
    for (int layer = 0; layer < DEPTH; layer++){
        for (int row = 0; row < input_size; row++){
            for (int col = 0; col < input_size; col++){
                printf("%u ", A[row][col+layer*input_size]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("Filter B = \n");
    for (int layer = 0; layer < DEPTH; layer++){
        for (int row = 0; row < filter_size; row++){
            for (int col = 0; col < filter_size; col++){
                printf("%u ", B[row][col+layer*filter_size]);
            }
            printf("\n");
        }
        printf("\n");
    }

    //compress 32 bits into 1 unsigned int
    unsigned int pop = filter_size*DEPTH;
    for (int i = 0; i < filter_size; i++) {
        for (int k = 0; k < comp_col; k++) {
            if(pop<=32) {
                for (int j = 0; j < pop; j++) {
                    xA[i][k] = xA[i][k] << 1;
                    xA[i][k] = xA[i][k] | A[i][j];
                }
            }
            else{
                for (int j = 0; j < 32; j++) {
                    xA[i][k] = xA[i][k] << 1;
                    xA[i][k] = xA[i][k] | A[i][j];
                }
                pop -= 32;
            }
        }
    }
    pop = filter_size*DEPTH;
    for (int i = 0; i < filter_size; i++) {
        for (int k = 0; k < comp_col; k++) {
            if(pop<=32) {
                for (int j = 0; j < pop; j++) {
                    xB[i][k] = xB[i][k] << 1;
                    xB[i][k] = xB[i][k] | B[i][j];
                }
            }
            else{
                for (int j = 0; j < 32; j++) {
                    xB[i][k] = xB[i][k] << 1;
                    xB[i][k] = xB[i][k] | B[i][j];
                }
                pop -= 32;
            }
        }
    }
//    gpuErrchk( cudaMemcpy(pA, a_f, pA_size, cudaMemcpyHostToDevice) );
//    gpuErrchk( cudaMemcpy(pB, b_f, pB_size, cudaMemcpyHostToDevice) );
//    gpuErrchk( cudaMemcpy(pC, c_f, pC_size, cudaMemcpyHostToDevice) );
    cudaMemcpy(tA, A, (input_size*input_size*DEPTH)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(tB, B, (filter_size*filter_size*DEPTH)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(tC, C, (output_size*output_size)*sizeof(unsigned int), cudaMemcpyHostToDevice);

    //    gpuErrchk( cudaMemcpy(fmap, i_fmap, fmap_size, cudaMemcpyHostToDevice) );
    //    gpuErrchk( cudaMemcpy(fil, filter, filter_size, cudaMemcpyHostToDevice) );
//    gpuErrchk( cudaMemcpy(result, result_return, int_size, cudaMemcpyHostToDevice) );
//    dim3 threadsPerBlock(NTHREADS_X,  1); //(int)std::ceil(NTHREADS_Y*(double)N/(32.0*32))
//    dim3 NumberofBlocks((int) std::ceil( (double)N/NTHREADS_X),
//                        (int) std::ceil( (double)N/(32*NTHREADS_Y)));
//    int result_host[1];
//    cudaMemcpyToSymbol(result_device, &result_host,sizeof(result_host), cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(1,1); //(int)std::ceil(NTHREADS_Y*(double)N/(32.0*32))
//    dim3 NumberofBlocks(1,1);
    dim3 NumberofBlocks(output_size,1);

    cudaEventRecord(start);

//    Mat_XNOR<<<NumberofBlocks,threadsPerBlock>>>(pA,pB,pC);
//    cudaDeviceSynchronize();
    convolve<<<NumberofBlocks,threadsPerBlock>>>(tA, tB, tC, padding, stride);
//    gpuErrchk( cudaDeviceSynchronize() );

//    gpuErrchk( cudaMemcpy(c_f, pC, pC_size, cudaMemcpyDeviceToHost) );
    cudaMemcpy(C, tC, (output_size*output_size)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
//    cudaMemcpyFromSymbol(&result_host, &result_device, sizeof(result_host), 0, cudaMemcpyDeviceToHost);
//    cudaMemcpyFromSymbol(result_host, result_device, sizeof(result_host), 0);
//    gpuErrchk( cudaMemcpy(result_return, result, int_size, cudaMemcpyDeviceToHost) );
//    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(stop) );
    gpuErrchk( cudaEventSynchronize(stop) );

    printf("C = \n");
//    for (int layer = 0; layer < DEPTH; layer++){
    for (int row = 0; row < output_size; row++) {
        for (int col = 0; col < output_size; col++) {
            printf("%u ", C[row][col]);
        }
        printf("\n");
    }
    printf("\n");
//    printf("result_host = %d\n", result_host[0]);

#ifdef __NO_OUTPUT
    // print Matrix A float
//    printf("A = \n");
//    for (i = 0; i < a_nlines; i++)
//    {
//        for (j = 0; j < pA_ncolumns; j++)
//        {
//            printf("%u ", a_f[i * pA_ncolumns + j]);
//        }
//        printf("\n");
//    }
//    printf("\n");
//    // print Matrix B float
//    printf("B = \n");
//    for (i = 0; i < b_nlines; i++)
//    {
//        for (j = 0; j < pB_ncolumns; j++)
//        {
//            printf("%u ", b_f[i * pB_ncolumns + j]);
//        }
//        printf("\n");
//    }
//    printf("\n");
//    // print Matrix C float
//    printf("C = \n");
//    for (i = 0; i < c_nlines; i++)
//    {
//        for (j = 0; j < pC_ncolumns; j++)
//        {
//            printf("%u ", C[i * pC_ncolumns + j]);
//        }
//        printf("\n");
//    }

#endif

#ifdef __TIME
    float milliseconds = 0;
    gpuErrchk( cudaEventElapsedTime(&milliseconds, start, stop) );
    printf("Time shared:");
    printf("%.5f\n", milliseconds);
#endif

//    free(a); free(b); free(c);

    cudaFree(tA);
    cudaFree(tB);
    cudaFree(tC);

    printf("\n");

    return 0;
}


