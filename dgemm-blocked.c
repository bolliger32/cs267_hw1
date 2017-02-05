const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* restrict C)
{
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k) 
    {
      double bkj = B[k+j*lda];
      for (int i = 0; i < M; ++i)
      {
          C[i+j*lda] += A[i+k*lda] * bkj;
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
  for (int i = 0; i < lda; i += BLOCK_SIZE)
  {
    int M = min (BLOCK_SIZE, lda-i);
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
      int N = min (BLOCK_SIZE, lda-j);
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
	    int K = min (BLOCK_SIZE, lda-k);

	/* Perform individual block dgemm */
	   do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
}
