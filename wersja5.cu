#define GRID_HEIGHT (MATRIX_SIZE / BLOCK_SIZE)
#define GRID_WIDTH (GRID_HEIGHT)


__global__ void matrixMulCUDA(float *C, float *A, float *B) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = MATRIX_SIZE * BLOCK_SIZE * by;
    int aEnd = aBegin + MATRIX_SIZE - 1;
    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * MATRIX_SIZE;

    float C_local = 0;

    // macierze na których wykonujemy obliczenia
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // macierze do których równolegle z obliczeniami wpisujemy dane kolejnych bloków
    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];

    A_shared[ty][tx] = A[aBegin + MATRIX_SIZE * ty + tx];
    B_shared[ty][tx] = B[bBegin + MATRIX_SIZE * ty + tx];

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        //Pobieranie danych
        As[ty][tx] = A_shared[ty][tx];
        Bs[ty][tx] = B_shared[ty][tx];


        __syncthreads();

        A_shared[ty][tx] = A[a + MATRIX_SIZE * ty + tx];
        B_shared[ty][tx] = B[b + MATRIX_SIZE * ty + tx];

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            //Kolumna razy wiersz
            C_local += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
        
    }

    int c = MATRIX_SIZE * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + MATRIX_SIZE * ty + tx] = C_local;
}

// wywołanie
dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
dim3 grid(GRID_WIDTH, GRID_HEIGHT);
matrixMulCUDA<<< grid, threads >>>(d_C, d_A, d_B);