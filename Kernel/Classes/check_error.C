#include "Error.h"
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

void check_error (const char* method)
{
  cudaThreadSynchronize ();

  cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    cerr << method << " cudaGetLastError="
         << cudaGetErrorString (error) << endl;

    throw Error (InvalidState, method, cudaGetErrorString (error));
  }
}

void check_error_stream (const char* method, cudaStream_t stream)
{
  if (!stream)
    check_error (method);
  else
  {
    cudaStreamSynchronize (stream);

    cudaError error = cudaGetLastError();
    if (error != cudaSuccess)
    {
      cerr << method << " cudaGetLastError="
           << cudaGetErrorString (error) << endl;

      throw Error (InvalidState, method, cudaGetErrorString (error));
    }
  }
}
