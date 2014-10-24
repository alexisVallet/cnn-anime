/**
 * Wraps libjpeg-turbo to provide fast jpeg decoding.
 */
#include <stdio.h>
#include <turbojpeg.h>

/**
 * Reads the header of a raw jpeg buffer to get image info.
 */
void read_header(char *input, int size, int *rows, int *cols) {
  tjhandle decomp;
  int subsamp;
  
  decomp = tjInitDecompress();
  tjDecompressHeader2(decomp, input, (long unsigned int)size, cols, rows, &subsamp);
  tjDestroy(decomp);
}

/**
 * Decodes a jpeg buffer into an output buffer. Assumes the output buffer has already
 * been properly allocated.
 */
void decode_jpeg(char *input, int size, char *output) {
  tjhandle decomp;
  int width;
  int height;
  int subsamp;
  
  decomp = tjInitDecompress();
  tjDecompressHeader2(decomp, input, (long unsigned int)size, &width, &height, &subsamp);
  tjDecompress2(decomp, input, (long unsigned int)size, output, width, 0, height, TJPF_RGB, TJFLAG_FASTDCT);
  tjDestroy(decomp);
}
