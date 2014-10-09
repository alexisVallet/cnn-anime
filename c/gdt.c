#include <float.h>
#include <string.h>
#include <stdio.h>

static double eps = 10E-5;

static float min(float f1, float f2) {
  return f1 > f2 ? f2 : f1;
}

static float max(float f1, float f2) {
  return f1 > f2 ? f1 : f2;
}

/**
 * Implementation of the generalized distance transform, generalized to
 * quadratic distances. Implemented in C as I couldn't get an efficient
 * implementation in Python or Cython.
 *
 * @param d coefficient for the distance function d(p,q) = d[0](p-q) + d[1](p-q)^2
 * @param f function to compute the distance transform of as an array.
 * @param n size of array f.
 * @param df output distance transform of f. Should be allocated to n elements prior
 *           to call.
 * @param arg output indexes corresponding to the argmax version of the gdt. Should be
 *            allocated to n elements prior to call.
 */
void gdt1D(float *d, float *f, int n, float *df, int *arg, int offset, 
	   double range) {
  /*
   * Please refer to Felzenszwalb, Huttenlocher, 2004 for a detailed
   * pseudo code of the algorithm - which the code mirrors, except for
   * a few exceptions.
   */
  int k = 0;
  int v[n];
  float z[n * 2];
  float s;
  float unbs;
  int qmvk;
  int q;
  int vk, fvk, fq;
  float eps = 10E-5;
  v[0] = 0;
  z[0] = -FLT_MAX;
  z[1] = FLT_MAX;
  
  for (q = 1; q < n; q++) {
    /* Intersection s generalized to arbitrary parabolas (d[1] nonnegative). 
     * Follows from elementary algebra.
     */
  inter: 
    /*
     * From the source code by Girshick, bounding the distance function
     * into a range - deals with the case where a=b=0 I guess. Real ugly
     * though, it looks like it's always going to converge to that.
     */
    vk = v[k];
    fvk = f[vk];
    fq = f[q];
    
    unbs = (d[0] * (vk - q) + d[1] * (q*q - vk*vk) + fq - fvk)
      / (2*d[1]*(q - v[k]));
    s = min(v[k]+range+eps, max(q-range-eps,unbs));

    if (s <= z[k]) {
      k--;
      goto inter;
    } else {
      k++;
      v[k] = q;
      z[k] = s;
      z[k+1] = FLT_MAX;
    }
  }

  k = 0;
  for (q = 0; q < n; q++) {
    while (z[k+1] < q) {
      k++;
    }
    /* Compared to the original paper, swapped in the new distance definition. */
    qmvk = q - v[k];
    vk = v[k];
    fvk = f[vk];
    df[q] = d[0] * qmvk + d[1] * qmvk * qmvk + fvk;
    /* Store the index of the actual max in the arg vector. Necessary for efficient
     * displacement lookup in the DPM matching algorithm. */
    arg[q] = offset + v[k];
  }
}

static int toRowMajor(int i, int j, int cols) {
  return i * cols + j;
}

static void fromRowMajor(int flat, int *i, int *j, int cols) {
  *i = flat / cols;
  *j = flat % cols;
}

static int toColMajor(int i, int j, int rows) {
  return i + j * rows;
}

static void fromColMajor(int flat, int *i, int *j, int rows) {
  *j = flat / rows;
  *i = flat % rows;
}

/**
 * Matrix transpose code from http://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c .
 */
void tran(void *src, void *dst, const int N, const int M, size_t coeffsize) {
  int n, i, j;

  for(n = 0; n<N*M; n++) {
    i = n/N;
    j = n%N;
    memcpy(dst + n * coeffsize, src + (M * j + i) * coeffsize, coeffsize);
  }
}

/**
 * Compute the 2D generalized distance transform of a function. All output arrays should
 * be allocated prior to the call.
 *
 * @param d 4 elements array indicating coefficients for the quadratic distance.
 * @param f function to compute the distance transform of, rows*cols row-major matrix.
 * @param rows the number of rows on the grid.
 * @param cols the number of columns on the grid.
 * @param df output rows*cols row-major matrix for the distance transform of f.
 * @param argi output row indexes for the argmax version of the problem. rows*cols
 *             row-major matrix.
 * @param argj output column indexes for the argmax version of the problem. rows*cols
 *             row-major matrix.
 */
void gdt2D(float *d, float *f, int rows, int cols,
	   float *df, int *argi, int *argj, int range) {
  // apply the 1D algorithm on each row
  int i;
  int j;
  float dx[2] = {d[0], d[2]};
  float dy[2] = {d[1], d[3]};
  int offset;
  float df2[rows * cols];
  float df3[rows * cols];
  int flatarg1[rows * cols];
  int flatarg2[rows * cols];
  int tmpi, tmpj;
  int flatidx1, flatidx2;
  int outIdx;

  for (i = 0; i < rows; i++) {
    offset = toRowMajor(i,0,cols);
    gdt1D(dy, f + offset, cols, df + offset, flatarg1 + offset, offset,
	  range);
  }

  // then on each column of the result. For this we transpose it, for memory locality.
  tran(df, df2, rows, cols, sizeof(float));
  
  for (i = 0; i < cols; i++) {
    offset = toRowMajor(i, 0, rows);
    gdt1D(dx, df2 + offset, rows, df3 + offset, flatarg2 + offset, offset,
	  range);
  }

  // transpose the result again
  tran(df3, df, cols, rows, sizeof(float));

  /* Compute 2D indexes of max, using the flat indexes of each pass.
   *
   * flatarg2 is a row-major cols*rows matrix. flatarg2[j,i] is flat index
   * into a row-major cols*rows matrix. Therefore we have to swap the resulting
   * indexes.
   * 
   * flatarg1 can be thought of as a row major rows*cols matrix
   * containing (i,j) row major indexes into f.
   */
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      flatidx1 = flatarg2[toRowMajor(j,i,rows)];
      fromRowMajor(flatidx1, &tmpj, &tmpi, rows);
      flatidx2 = flatarg1[toRowMajor(tmpi, tmpj, cols)];
      outIdx = toRowMajor(i,j,cols);
      fromRowMajor(flatidx2, &argi[outIdx], &argj[outIdx], cols);
    }
  }
}
