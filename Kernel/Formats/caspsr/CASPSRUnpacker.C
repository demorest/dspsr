//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CASPSRUnpacker.h"
#include "dsp/BitTable.h"

#include "Error.h"

#if HAVE_CUFFT
#include "MemoryCUDA.h"
#include "dsp/CASPSRUnpackerCUDA.h"
#endif

using namespace std;

dsp::CASPSRUnpacker::CASPSRUnpacker (const char* _name) : HistUnpacker (_name)
{
  if (verbose)
    cerr << "dsp::CASPSRUnpacker ctor" << endl;

  set_nstate (256);
  on_gpu = false;

  table = new BitTable (8, BitTable::TwosComplement);
}

//! Return true if the unpacker can operate on the specified device
bool dsp::CASPSRUnpacker::get_device_supported (Memory* memory) const
{
#if HAVE_CUFFT
  return dynamic_cast< CUDA::Memory*> ( memory );
#else
  return false;
#endif
}

//! Set the device on which the unpacker will operate
void dsp::CASPSRUnpacker::set_device (Memory*)
{
#if HAVE_CUFFT
  on_gpu = dynamic_cast< CUDA::Memory*> ( memory );
#else
  throw Error (InvalidState, "dsp::CASPSRUnpacker::set_device",
	       "unsupported device");
#endif
}


bool dsp::CASPSRUnpacker::matches (const Observation* observation)
{
  return observation->get_machine()== "CASPSR"
    && observation->get_nbit() == 8;
}

void dsp::CASPSRUnpacker::unpack (uint64_t ndat,
				  const unsigned char* from,
				  const unsigned nskip,
				  float* into,
				  const unsigned fskip,
				  unsigned long* hist)
{
  const float* lookup = table->get_values ();
  int counter_four = 0;

  if (verbose)
    cerr << "dsp::CASPSRUnpacker::unpack ndat=" << ndat << endl;

  for (uint64_t idat = 0; idat < ndat; idat++)
  {
    hist[ *from ] ++;
    *into = lookup[ *from ];
    
#ifdef _DEBUG
    cerr << idat << " " << int(*from) << "=" << *into << endl;
#endif
    counter_four++;
    if (counter_four == 4)
      {
	counter_four = 0;
	from += 5; //(nskip+4);
      }
    else
      {
	from ++; //=nskip;
      }
    into += fskip;
  }
}

void dsp::CASPSRUnpacker::unpack ()
{
#if HAVE_CUFFT
  if (on_gpu)
  {
    unpack_on_gpu ();
    return;
  }
#endif

  const uint64_t   ndat  = input->get_ndat();
  
  const unsigned nchan = input->get_nchan();
  const unsigned npol  = input->get_npol();
  const unsigned ndim  = input->get_ndim();
  
  const unsigned nskip = npol * nchan * ndim;
  const unsigned fskip = ndim;
  
  unsigned offset = 0;
  
  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      if (ipol==1)
	offset = 4;
      for (unsigned idim=0; idim<ndim; idim++)
      {
	const unsigned char* from = input->get_rawptr() + offset;
	float* into = output->get_datptr (ichan, ipol) + idim;
	unsigned long* hist = get_histogram (ipol);
	      
#ifdef _DEBUG
	cerr << "c=" << ichan << " p=" << ipol << " d=" << idim << endl;
#endif
	      
	unpack (ndat, from, nskip, into, fskip, hist);
	offset ++;
      }
    }
  }
}

#if HAVE_CUFFT

void dsp::CASPSRUnpackerSetup::unpack_on_gpu ()
{
  const uint64_t ndat = input->get_ndat();

  staging.Observation::operator=( *input );
  staging.resize(ndat);

  // staging buffer on the GPU for packed data
  unsigned char* stagingBufGPU = staging.get_rawptr();
  //float* from_tmp;
  //float* from_tmp_cpu;
  //float* out_tmp;
  //float* out_tmp_cpu;

  //from_tmp_cpu = (float*) malloc(ndat*2*sizeof(float));
  //out_tmp_cpu = (float*) malloc(ndat*2*sizeof(float));
  

  const unsigned char*  from= input->get_rawptr();

  float* into_pola = output->get_datptr(0,0);
  float* into_polb = output->get_datptr(0,1);

  //uint64_t dataSize = ndat;
  int dimBlockUnpack(256);
  int dimGridUnpack(ndat / (dimBlockUnpack*4)); 

  if (dimBlockUnpack*dimGridUnpack*4 != ndat)
  {
     cerr << "dsp::CASPSRUnpackerSetup::unpack increasing dimGridUnpack by 1" << endl;
     dimGridUnpack = dimGridUnpack + 1;
  }
  cutilSafeCall(cudaMemcpy(stagingBufGPU,from,ndat*2*(sizeof(unsigned char)),cudaMemcpyHostToDevice));

  ///////////////////////

  //cerr << "casting input...";
  
  //for (uint i=0;i<ndat*2;i++)
  // {
      //cerr << (float) from[i] << endl;
      //  from_tmp_cpu[i] = 1; // (float) from[i];
      //cerr << ".." << i;
      //if (from_tmp_cpu[i] < 0 || from_tmp_cpu[i] > 255 || from_tmp_cpu[i] != int(from_tmp_cpu[i]))
      //cerr << "SOMETHING BADLY WRONG!!!" << endl;
  //}

  

  //cutilSafeCall(cudaMalloc((void**) &from_tmp,ndat*2*sizeof(float)));
  //cutilSafeCall(cudaMemset(from_tmp,0,ndat*2*sizeof(float)));

  //cerr << "...cudaMalloc...";

  //cutilSafeCall(cudaMemcpy(from_tmp,from_tmp_cpu,ndat*2*sizeof(float),cudaMemcpyHostToDevice));

  //cerr << "cudaMemcpy" << endl;
  

  //cutilSafeCall(cudaMalloc((void**) &out_tmp,ndat*2*sizeof(float)));
  //cutilSafeCall(cudaMemset(out_tmp,0,ndat*2*sizeof(float)));

  ///////////////////
  //cutilSafeCall(cudaMemcpy(tmp_copy,stagingBufGPU,ndat*2,cudaMemcpyDeviceToHost));

  cudaError error = cudaGetLastError();
  if (error != cudaSuccess)
    cerr << "CASPSRUnpackerSetup::unpack() cudaMemcpy FAIL: " << cudaGetErrorString (error) << endl;

  // buffer on gpu for unpacked data
  // should be malloc earlier along with the stagin buf? just need the pointers getdatprt? 
  //cutilSafeCall(cudaMalloc((void **) &unpackBufGPU,ndat*sizeof(cufftReal)));

  //call function
  //int device_no;
  //cudaGetDevice(&device_no);
  //cerr << "cuda device: " << device_no << endl;
  //cerr << "dsp::CASPSRUnpackerSetup::unpack() calling caspsr_unpack" << endl;
  //cerr << "from = " << &from << " input = " << &stagingBufGPU << " output [0] = " << into_pola << " output[1] = " << into_polb << endl;


  caspsr_unpack(ndat,stagingBufGPU,dimBlockUnpack,dimGridUnpack,into_pola,into_polb);

  //cutilSafeCall(cudaMemcpy(out_tmp_cpu,out_tmp,ndat*2*sizeof(float),cudaMemcpyDeviceToHost));

  //for (int i=0;i<ndat*2;i++)
  //  {
  //    if (from_tmp_cpu[i] != out_tmp_cpu[i])
  //	cerr << "ndat :" << i << " from " << from_tmp_cpu[i] <<" NOT EQUAL " << out_tmp_cpu[i] << endl;
      //else 
      //	 cerr << "ndat :" << i << " from " << from_tmp_cpu[i] << " IS EQUAL " << out_tmp_cpu[i] << endl;
  //  }
  
  //cerr << "dsp::CASPSRUnpackerSetup::unpack() end of not equal test" << endl;

  //cudaFree(out_tmp);
  //cudaFree(from_tmp);
  //free(out_tmp_cpu);
  //free(from_tmp_cpu);
}

#endif
