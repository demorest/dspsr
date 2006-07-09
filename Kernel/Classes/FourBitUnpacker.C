/***************************************************************************
 *
 *   Copyright (C) 2004 by Stephen Ord
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <iostream>
#include <assert.h>
#include <math.h>

#include "dsp/FourBitTable.h"
#include "dsp/FourBitUnpacker.h"

#include "Error.h"


//! Null constructor
dsp::FourBitUnpacker::FourBitUnpacker (const char* _name) : Unpacker (_name)
{
  nsamples = 512;
}

dsp::FourBitUnpacker::~FourBitUnpacker ()
{
}

void dsp::FourBitUnpacker::set_table (FourBitTable* _table)
{
  if (table.get() == _table)
    return;

  if (verbose)
    cerr << "dsp::FourBitUnpacker::set_table" << endl;

  table = _table;
}

const dsp::FourBitTable* dsp::FourBitUnpacker::get_table () const
{ 
  return table;
}

//! Initialize and resize the output before calling unpack
void dsp::FourBitUnpacker::transformation ()
{
  if (verbose)
    cerr << "dsp::FourBitUnpacker::transformation" << endl;;

  // set the Observation information
  output->Observation::operator=(*input);
  if (verbose) {
    cerr << "dsp::FourBitUnpacker::transformation output ndat= " << input->get_ndat() << " npol = " << output->get_npol() << " nchan = " << output->get_nchan() << endl;
  }
  // resize the output 
  output->resize (input->get_ndat());


  if( verbose )
    fprintf(stderr,"dsp::FourBitUnpacker::transformation() calling unpack()\n");

  // unpack the data
  unpack ();

  if (verbose)
    cerr << "dsp::FourBitUnpacker::transformation exit" << endl;
}

void dsp::FourBitUnpacker::unpack ()
{
  uint64 ndat = input->get_ndat();

  unsigned samples_per_byte = 2;

  if (ndat % samples_per_byte)
    throw Error (InvalidParam, "dsp::FourBitUnpacker::check_input",
		 "input ndat="I64" != %dn", ndat, samples_per_byte);
  

  const unsigned char* rawptr = input->get_rawptr();

  const unsigned char* from = rawptr;

  //unsigned ipol = get_output_ipol ();
  //int nchan = output->get_nchan();
 
  if (verbose)
    cerr << "dsp::FourBitUnpacker::unpack into simple unpack";

   simple_unpack (from, ndat);
      
  if (verbose)
    cerr << "dsp::FourBitUnpacker::unpack out of  simple unpack";

  output->seek (input->get_request_offset());
  output->set_ndat (input->get_request_ndat());


    
}

void dsp::FourBitUnpacker::simple_unpack (
					const unsigned char* input_data, 
					uint64 ndat)
{

  unsigned samples_per_byte = 2;

   
  const unsigned char * input_data_ptr = input_data;
  float  * output_data = NULL;
  unsigned val;
  int nchan = output->get_nchan();
  unsigned bytes = input->get_ndat()*nchan/samples_per_byte;
  int tsamp = 0;
  //float *thr = output->get_thresh();
 
  if (verbose){
    fprintf(stderr,"input_data=%p\n",input_data);
    fprintf(stderr,"ndat="UI64"\n",ndat);
    fprintf(stderr,"nchan=%d\n",nchan);
  }
 
  const float *twovals = 0;

 
  for (unsigned bt=0; bt<bytes; bt++) {
    val = unsigned(*input_data_ptr);
    twovals = table->get_two_vals(val);
    tsamp = (bt*samples_per_byte)/nchan;    
    for (unsigned pt=0; pt<samples_per_byte; pt++) {
      // cout << "timesample " << (numtsamps + (bt*samples_per_byte)/nchan) << " byte " << bt << " channel " << (((bt*2)%nchan)+pt) << " value " << twovals[pt] << endl;
      output_data = output->get_datptr((((bt*2)%nchan)+pt),0) + tsamp;
      *output_data = twovals[pt] ;
    }
    input_data_ptr += 1;
  }
  
}

int64 dsp::FourBitUnpacker::stats(vector<double>& _sum, vector<double>& _sumsq) {
  
  if (!input)
    throw Error (InvalidState, "dsp::FourBitCorrection::stats", "no input");

  if (input->get_nbit() != 4)
    throw Error (InvalidParam, "dsp::FourBitCorrection::stats",
		 "input nbit != 4");

  if (input->get_state() != Signal::Nyquist)
    throw Error (InvalidParam, "dsp::FourBitCorrection::stats",
		 "input state != Nyquist");

  unsigned samples_per_byte = 2;

  const unsigned char * input_data_ptr = input->get_rawptr();
  
if (_sum.size() > 1) {
    cerr << "FourBitUnpacker::stats more than one digitiser sum requested: only one supplied\n";
  }
  // samples per byte = 2
  unsigned bytes = nsamples/samples_per_byte;

  const float *twovals = 0;

  double sum = 0.0;
  double sumsq = 0.0;
  double value = 0.0;
  unsigned val;
  unsigned total_samples = 0;
 
  for (unsigned bt=0; bt<bytes; bt++) {
    val = unsigned(*input_data_ptr);
    twovals = table->get_two_vals(val);
    for (unsigned pt=0; pt<samples_per_byte; pt++) {
      value = twovals[pt];
      sum = sum + value;
      sumsq = sumsq + (value*value);
    }
    input_data_ptr += 1;
    total_samples = total_samples + 2;
  }

  if (verbose) {
    cerr << "FourBitUnpacker::stats sum " << sum << " sumsq: " << sumsq << endl;
  } 
  _sum[0] = sum;
  _sumsq[0] = sumsq;

  return total_samples;     
}

unsigned dsp::FourBitUnpacker::get_output_ipol() {
  return 0;
}

