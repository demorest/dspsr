#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"

#include "dsp/TimeOrder.h"

dsp::TimeOrder::TimeOrder() : Transformation<TimeSeries,BitSeries>("TimeOrder",outofplace){ }

dsp::TimeOrder::~TimeOrder(){ }

void dsp::TimeOrder::transformation(){
  output->Observation::operator=( *input );
  output->resize( input->get_ndat() );

  // number of floats between (t0,f0) and (t1,f0) of a BitSeries
  register const unsigned output_stride = input->nbytes(1)/8;

  for( unsigned ichan=0; ichan<input->get_nchan(); ichan++){
    for( unsigned ipol=0; ipol<input->get_npol(); ipol++){
      float* in = input->get_datptr(ichan,ipol);
      float* out = (float*)output->get_rawptr();

      register const unsigned nsamp = input->get_ndat(); 
      register unsigned output_samp=0;

      for( unsigned isamp=0; isamp<nsamp;
	   isamp++, output_samp+=output_stride)
	out[output_samp] = in[isamp];
    }
  }
 
}
