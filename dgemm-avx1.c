
#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(M_BLOCK_SIZE)
#define M_BLOCK_SIZE 32
#endif

#if !defined(N_BLOCK_SIZE)
#define N_BLOCK_SIZE 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void inline do_block (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C)
{
  __m256d c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, b2, b3, t0, t1, t2, t3;
  
  /* For each row i of A */
  for (int i = 0; i < M/4*4; i += 4)
  {
    /* For each column j of B */ 
    for (int j = 0; j < N/4*4; j += 4)
    {
      c0 = _mm256_loadu_pd(C + i + j*lda);
      c1 = _mm256_loadu_pd(C + i + (j+1)*lda);
      c2 = _mm256_loadu_pd(C + i + (j+2)*lda);
      c3 = _mm256_loadu_pd(C + i + (j+3)*lda);
      /* Compute C(i,j) */
      for (int k = 0; k < K/4*4; k += 4)
      {
        a0 = _mm256_loadu_pd(A + i + k*lda);
        a1 = _mm256_loadu_pd(A + i + (k+1)*lda);
        
        b0 = _mm256_broadcast_sd(B + j*lda + k);
        b1 = _mm256_broadcast_sd(B + (j+1)*lda + k);
        b2 = _mm256_broadcast_sd(B + (j+2)*lda + k);
        b3 = _mm256_broadcast_sd(B + (j+3)*lda + k);
        
        t0 = _mm256_mul_pd(a0, b0);
        t1 = _mm256_mul_pd(a0, b1);
        t2 = _mm256_mul_pd(a0, b2);
        t3 = _mm256_mul_pd(a0, b3);
        
        c0 = _mm256_add_pd(c0, t0);
        c1 = _mm256_add_pd(c1, t1);
        c2 = _mm256_add_pd(c2, t2);
        c3 = _mm256_add_pd(c3, t3);

        b0 = _mm256_broadcast_sd(B + j*lda + (k+1));
        b1 = _mm256_broadcast_sd(B + (j+1)*lda + (k+1));
        b2 = _mm256_broadcast_sd(B + (j+2)*lda + (k+1));
        b3 = _mm256_broadcast_sd(B + (j+3)*lda + (k+1));

        t0 = _mm256_mul_pd(a1, b0);
        t1 = _mm256_mul_pd(a1, b1);
        t2 = _mm256_mul_pd(a1, b2);
        t3 = _mm256_mul_pd(a1, b3);
        
        c0 = _mm256_add_pd(c0, t0);
        c1 = _mm256_add_pd(c1, t1);
        c2 = _mm256_add_pd(c2, t2);
        c3 = _mm256_add_pd(c3, t3);
        
        a2 = _mm256_loadu_pd(A + i + (k+2)*lda);
        a3 = _mm256_loadu_pd(A + i + (k+3)*lda);
        
        b0 = _mm256_broadcast_sd(B + j*lda + (k+2));
        b1 = _mm256_broadcast_sd(B + (j+1)*lda + (k+2));
        b2 = _mm256_broadcast_sd(B + (j+2)*lda + (k+2));
        b3 = _mm256_broadcast_sd(B + (j+3)*lda + (k+2));

        t0 = _mm256_mul_pd(a2, b0);
        t1 = _mm256_mul_pd(a2, b1);
        t2 = _mm256_mul_pd(a2, b2);
        t3 = _mm256_mul_pd(a2, b3);
        
        c0 = _mm256_add_pd(c0, t0);
        c1 = _mm256_add_pd(c1, t1);
        c2 = _mm256_add_pd(c2, t2);
        c3 = _mm256_add_pd(c3, t3);

        b0 = _mm256_broadcast_sd(B + j*lda + (k+3));
        b1 = _mm256_broadcast_sd(B + (j+1)*lda + (k+3));
        b2 = _mm256_broadcast_sd(B + (j+2)*lda + (k+3));
        b3 = _mm256_broadcast_sd(B + (j+3)*lda + (k+3));
        
        t0 = _mm256_mul_pd(a3, b0);
        t1 = _mm256_mul_pd(a3, b1);
        t2 = _mm256_mul_pd(a3, b2);
        t3 = _mm256_mul_pd(a3, b3);
        
        c0 = _mm256_add_pd(c0, t0);
        c1 = _mm256_add_pd(c1, t1);
        c2 = _mm256_add_pd(c2, t2);
        c3 = _mm256_add_pd(c3, t3);
      }
      for (int k = K/4*4; k < K; ++k)
      {
        a0 = _mm256_loadu_pd(A + i + k*lda);
        
        b0 = _mm256_broadcast_sd(B + j*lda + k);
        b1 = _mm256_broadcast_sd(B + (j+1)*lda + k);
        b2 = _mm256_broadcast_sd(B + (j+2)*lda + k);
        b3 = _mm256_broadcast_sd(B + (j+3)*lda + k);

        t0 = _mm256_mul_pd(a0, b0);
        t1 = _mm256_mul_pd(a0, b1);
        t2 = _mm256_mul_pd(a0, b2);
        t3 = _mm256_mul_pd(a0, b3);
        
        c0 = _mm256_add_pd(c0, t0);
        c1 = _mm256_add_pd(c1, t1);
        c2 = _mm256_add_pd(c2, t2);
        c3 = _mm256_add_pd(c3, t3);
      }
      _mm256_storeu_pd(C + i + j*lda, c0);
      _mm256_storeu_pd(C + i + (j+1)*lda, c1);
      _mm256_storeu_pd(C + i + (j+2)*lda, c2);
      _mm256_storeu_pd(C + i + (j+3)*lda, c3);
    }
    for (int j = N/4*4; j < N; ++j)
    {
      c0 = _mm256_loadu_pd(C + i + j*lda);
      for (int k = 0; k < K; ++k)
      {
        a0 = _mm256_loadu_pd(A + i + k*lda);
        b0 = _mm256_broadcast_sd(B + j*lda + k);
        c0 = _mm256_add_pd(c0, _mm256_mul_pd(a0, b0));
      }
      _mm256_storeu_pd(C + i + j*lda, c0);
    }
  }
  
  
  for (int i = M/4*4; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      for (int k = 0; k < K; ++k)
      {
        C[i+j*lda] += A[i+k*lda] * B[k+j*lda];
      }
    }
  }

}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm (int lda, double* A, double* B, double* restrict C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += M_BLOCK_SIZE)
  {
    int M = min (M_BLOCK_SIZE, lda-i);
    /* For each block-column of B */
    for (int j = 0; j < lda; j += M_BLOCK_SIZE)
    {
      int N = min (M_BLOCK_SIZE, lda-j);
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += N_BLOCK_SIZE)
      {
	    int K = min (N_BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	   do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
}

