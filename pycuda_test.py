import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from string import Template

from pycuda.compiler import SourceModule

a = 4
N = 4
input_size = 4
filter_size = 3
depth = 1

mod1 = Template("""
    #define M ${a}
    __global__ void Mat_XNOR(unsigned int A[][M], unsigned int B[][M], unsigned int C[][M]){
    int NTHREADS_X = 32;
    int NTHREADS_Y = 32;
    int N = M;
    int m_size=M;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int line =  blockIdx.y * blockDim.y + threadIdx.y;

    __syncthreads();

    if (m_size % NTHREADS_Y != 0){
    int edge_size = m_size % NTHREADS_Y;
    if((blockIdx.x==(int) std::ceil( (double)N/NTHREADS_X))
    && (blockIdx.y==(int) std::ceil( (double)N/NTHREADS_Y))){
    for(int i=0; i<edge_size; i++) {
    column = (blockIdx.x - 1) * blockDim.x + i;
    line = (blockIdx.y - 1) * blockDim.y + i;
    C[column][line] = ~((A[column][line] ^ B[column][line]));
    //C[column][line] = ((A[column][line] * B[column][line]));
    }
    }
    else if ((blockIdx.x==(int) std::ceil( (double)N/NTHREADS_X))
    && (blockIdx.y!=(int) std::ceil( (double)N/NTHREADS_Y))){
    for(int i=0; i<edge_size; i++) {
    column = (blockIdx.x - 1) * blockDim.x + i;
    line =  blockIdx.y * blockDim.y + threadIdx.y;
    C[column][line] = ~((A[column][line] ^ B[column][line]));
    //C[column][line] = ((A[column][line] * B[column][line]));
    }
    }
    else if ((blockIdx.x!=(int) std::ceil( (double)N/NTHREADS_X))
    && (blockIdx.y==(int) std::ceil( (double)N/NTHREADS_Y))){
    for(int i=0; i<edge_size; i++) {
    column = blockIdx.x * blockDim.x + threadIdx.x;
    line = (blockIdx.y - 1) * blockDim.y + i;
    C[column][line] = ~((A[column][line] ^ B[column][line]));
    //C[column][line] = ((A[column][line] * B[column][line]));
    }
    }
    }
    if ((m_size % NTHREADS_Y == 0)
    || (blockIdx.x!=(int) std::ceil( (double)N/NTHREADS_X))
    || (blockIdx.y!=(int) std::ceil( (double)N/NTHREADS_Y))) {
    C[column][line] = ~((A[column][line] ^ B[column][line]));
    //C[column][line] = ((A[column][line] * B[column][line]));
    }
    
    __syncthreads();
    }
    """)

mod2 = Template("""
    #define N ${N}
    __global__ void Mat_ACC(unsigned int C[][N/32], int c_nlines, int pC_ncolumns, int filter_size, int result[]){
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int line =  blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int counter = 0;
    __shared__ int total;

    unsigned int threshold = filter_size*filter_size/2;
    __syncthreads();
    counter += __popc(C[column][line]);
    atomicAdd(&total, counter);
    __syncthreads();
    if (total > threshold){
    result[0] = 1;
    }
    else{
    result[0] = 0;
    }
    }
    """)

mod3 = Template("""
#define input_size ${input_size}
#define filter_size ${filter_size}
#define depth ${depth}
__global__ void conv(unsigned int i_fmap[input_size][input_size][depth],
                     unsigned int filter [filter_size][filter_size][depth], int padding, int stride) {

    int output = ((input_size - filter_size + (2 * padding)) / stride) + 1;

    unsigned int padded_ifmap[input_size][input_size][depth];
    for (int layer = 0; layer < depth; layer++) {
        for (int vertical = 0; vertical < input_size + 2 * padding; vertical++) {
            for (int horizontal = 0; horizontal < input_size + 2 * padding; horizontal++) {
                if ((horizontal >= padding) && (vertical >= padding) &&
                    (horizontal <= (input_size + 2 * padding - 1 - padding)) &&
                    (vertical <= (input_size + 2 * padding - 1 - padding))) {

                    padded_ifmap[horizontal][vertical][layer] = i_fmap[horizontal - padding][vertical - padding][layer];
                } else {
                    //pad with 0
                    padded_ifmap[horizontal][vertical][layer] = 0;
                }
            }
        }
    }

    //when the input matrix is padded
    for (int layer = 0; layer < depth; layer++) {

        //copying filter for B
        unsigned int B[filter_size][filter_size];
        for (int row = 0; row < filter_size; row++) {
            for (int col = 0; col < filter_size; col++) {
                B[row][col] = filter[row][col][layer];
            }
        }

        for (int vertical = 0; vertical < output; vertical++) {
            for (int horizontal = 0; horizontal < output; horizontal++) {

                int start_idx_row = horizontal * stride;
                int start_idx_col = vertical * stride;
                unsigned int A[filter_size][filter_size];

                for (int row = 0; row < filter_size; row++) {
                    for (int col = 0; col < filter_size; col++) {
                        A[row][col] = padded_ifmap[start_idx_row + row][start_idx_col + col][layer];
                    }
                }

                unsigned int C[filter_size][filter_size];

                //Mat_XNOR<<<NumberofBlocks,threadsPerBlock>>>(A,B,C);

            }
        }
    }
}
    """)


mod_XNOR = SourceModule(mod1.substitute(a=a))
mod_ACC = SourceModule(mod2.substitute(N=N))
mod_conv = SourceModule(mod3.substitute(filter_size=filter_size, depth=depth, input_size=input_size))

Mat_XNOR = mod_XNOR.get_function("Mat_XNOR")
Mat_ACC = mod_ACC.get_function("Mat_ACC")
conv = mod_conv.get_function("conv")
#a_f = numpy.random.randn((1,1), dtype=numpy.uint32)
a_f = np.random.choice([0, 1], size=(N,N))
print(a_f)
print('\n')
#b_f = numpy.random.randn((1,1), dtype=numpy.uint32)
b_f = np.random.choice([0, 1], size=(N,N))
print(b_f)
print('\n')
#c_f = numpy.random.randn((1,1), dtype=numpy.uint32)
c_f = np.zeros((N,N))
print(c_f)
print('\n')
a_f = a_f.astype(np.uint32)
b_f = b_f.astype(np.uint32)
c_f = c_f.astype(np.uint32)

pA = cuda.mem_alloc(a_f.nbytes)
pB = cuda.mem_alloc(b_f.nbytes)
pC = cuda.mem_alloc(c_f.nbytes)
cuda.memcpy_htod(pA, a_f)
cuda.memcpy_htod(pB, b_f)
cuda.memcpy_htod(pC, c_f)
Mat_XNOR(pA, pB, pC, block=(N,N,1))
cuda.memcpy_dtoh(c_f, pC)

print(c_f)
print(c_f.shape)
print('\n')


