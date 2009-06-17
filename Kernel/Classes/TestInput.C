/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/TestInput.h"
#include "dsp/Input.h"
#include "dsp/BitSeries.h"

using namespace std;

bool dsp::TestInput::verbose = false;

//! Constructor
dsp::TestInput::TestInput ()
{
  block_size = 4096;
}

//! Destructor
dsp::TestInput::~TestInput ()
{
}
    
//! Run test using two Input instances that refer to the same data
void dsp::TestInput::runtest (Input* input_small, Input* input_large)
{
  if (verbose)
    cerr << "dsp::TestInput::runtest Creating BitSeries instances" << endl;

  Reference::To<dsp::BitSeries> data_small = new dsp::BitSeries;
  Reference::To<dsp::BitSeries> data_large = new dsp::BitSeries;

  cerr << "dsp::TestInput::runtest Input=" << input_large->get_name() << endl;

  unsigned resolution = input_small->get_resolution();

  cerr << "dsp::TestInput::runtest resolution = " << resolution << endl;

  if (resolution == 1)
    cerr << "dsp::TestInput::runtest WARNING: "
	"cannot fully test Input class" << endl;
    
  input_small->set_output (data_small);
  input_large->set_output (data_large);

  unsigned small_block = unsigned(block_size);

  // ensure that the small block size triggers resolution-related code
  unsigned modres = small_block % resolution;
  
  if (modres == 0) {
    small_block --;
    modres = resolution - 1;
  }
  
  unsigned large_block = small_block * resolution;
  
  cerr << "dsp::TestInput::runtest small block size = " << small_block << endl;
  cerr << "dsp::TestInput::runtest large block size = " << large_block << endl;

  input_small->set_block_size (small_block);
  input_large->set_block_size (large_block);

  unsigned block = 0;

  errors = 0;

  while (!input_large->eod()) {

    input_large->operate();

    // test that Input produces the expected output
    if (data_large->get_ndat() != large_block)
      cerr << "WARNING: block=" << block << " Input::block_size="<< large_block 
	   << " != BitSeries::ndat=" << data_large->get_ndat() << endl;
    
    if (data_large->get_request_ndat() != data_large->get_ndat()) {
      cerr << "ERROR: block=" << block << " large BitSeries::request_ndat=" 
           << data_large->get_request_ndat() << " != BitSeries::ndat=" 
	   << data_large->get_ndat() << endl;
      errors ++;
    }

    if (data_large->get_request_offset() != 0) {
      cerr << "ERROR: block=" << block << " large BitSeries::request_offset!=0"
           << endl;
      errors ++;
    }
    
    for (unsigned ismall=0; ismall<resolution; ismall++) {

      if (input_small->eod())
        break;

      input_small->operate();
      
      if (data_small->get_request_ndat() != small_block) {
	cerr << "WARNING: block=" << block << " small Input::block_size=" 
             << small_block << " != BitSeries::request_ndat=" 
	     << data_small->get_request_ndat() << endl;
      }
      
      uint64_t expected_offset = (ismall * modres) % resolution;
      
      if (data_small->get_request_offset() != expected_offset) {
	cerr << "ERROR: block=" << block << " BitSeries::request_offset="
	     << data_small->get_request_offset() << " != expected offset="
	     << expected_offset << endl;
	errors ++;
      }
      
      // on the first read of each loop, the first small_block samples
      // of data_small should equal those of data_large
      if (ismall == 0) {
	
	unsigned char* bytes_small = data_small->get_rawptr();
	unsigned char* bytes_large = data_large->get_rawptr();
	
	uint64_t nbyte = data_small->get_nbytes();
	if (nbyte > data_large->get_nbytes())
          nbyte = data_large->get_nbytes();

        if (data_small->get_start_time() != data_large->get_start_time()) {
           cerr << "ERROR: small.start_time=" << data_small->get_start_time()
                << " != large.start_time=" << data_large->get_start_time() 
                << endl;
           errors ++;
        }

	for (unsigned ibyte=0; ibyte < nbyte; ibyte++) {
	  if (bytes_small[ibyte] != bytes_large[ibyte]) {
	    fprintf (stderr, "ERROR: block=%d data[%d] small=%x != large=%x\n",
		     block, ibyte, bytes_small[ibyte], bytes_large[ibyte]);
	    errors ++;
	  }
	}
      }
    }
    block ++;
  }
} 
