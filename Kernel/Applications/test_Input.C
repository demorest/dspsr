#include <iostream>
#include <unistd.h>

#include "dsp/File.h"
#include "dsp/BitSeries.h"

#include "string_utils.h"
#include "dirutil.h"
#include "Error.h"

static char* args = "b:t:vV";

void usage ()
{
  cout << "test_Input - test time sample resolution features of Input class\n"
    "Usage: test_Input [" << args << "] file1 [file2 ...] \n"
    " -b block size  the base block size used in the test\n"
    " -t blocks      (stop before the end of the file)\n"
       << endl;
}

int main (int argc, char** argv) 

{ try {

  char* metafile = 0;
  bool verbose = false;

  int blocks = 0;
  unsigned block_size = 4096;

  int c;
  while ((c = getopt(argc, argv, args)) != -1)
    switch (c) {

    case 'V':
      dsp::Observation::verbose = true;
      dsp::Operation::verbose = true;

    case 'v':
      verbose = true;
      break;

    case 'b':
      block_size = atoi (optarg);
      break;

    case 't':
      blocks = atoi (optarg);
      break;

    default:
      cerr << "invalid param '" << c << "'" << endl;
    }

  vector <string> filenames;

  if (metafile)
    stringfload (&filenames, metafile);
  else 
    for (int ai=optind; ai<argc; ai++)
      dirglob (&filenames, argv[ai]);

  if (filenames.size() == 0) {
    usage ();
    return 0;
  }

  if (verbose)
    cerr << "Creating BitSeries instances" << endl;
  Reference::To<dsp::BitSeries> data_small = new dsp::BitSeries;
  Reference::To<dsp::BitSeries> data_large = new dsp::BitSeries;

  Reference::To<dsp::Input> input_small;
  Reference::To<dsp::Input> input_large;

  for (unsigned ifile=0; ifile < filenames.size(); ifile++) try {

    if (verbose)
      cerr << "opening data file " << filenames[ifile] << endl;

    input_small = dsp::File::create (filenames[ifile]);
    input_large = dsp::File::create (filenames[ifile]);

    if (verbose)
      cerr << "data file " << filenames[ifile] << " opened" << endl;

    unsigned resolution = input_small->get_resolution();

    if (resolution == 1)
      cerr << "WARNING: time sample resolution == 1. "
	"cannot fully test Input class" << endl;
    
    input_small->set_output (data_small);
    input_large->set_output (data_large);

    unsigned small_block = block_size;

    if (block_size % resolution)
      small_block --;

    unsigned large_block = small_block * resolution;

    input_small->set_block_size (small_block);
    input_large->set_block_size (large_block);
      
    int block=0;

    while (!input_large->eod()) {

      input_large->operate();

      // test that Input produces the expected output
      if (data_large->get_ndat() != large_block)
	cerr << "ERROR: Input::block_size=" << large_block 
	     << " != BitSeries::ndat=" << data_large->get_ndat() << endl;

      if (data_large->get_request_ndat() != large_block)
	cerr << "ERROR: large Input::block_size=" << large_block 
	     << " != BitSeries::request_ndat=" 
	     << data_large->get_request_ndat() << endl;

      if (data_large->get_request_offset() != 0)
	cerr << "ERROR: BitSeries::request_offset != 0 [large]" << endl;

      for (unsigned ismall=0; ismall<resolution; ismall++) {

	input_small->operate();

	if (data_small->get_request_ndat() != small_block)
	  cerr << "ERROR: small Input::block_size=" << small_block 
	       << " != BitSeries::request_ndat=" 
	       << data_small->get_request_ndat() << endl;

	uint64 expected_offset = (resolution - ismall) % resolution;

	if (data_small->get_request_offset() != expected_offset)
	  cerr << "ERROR: BitSeries::request_offset="
	       << data_small->get_request_offset() << " != expected offset="
	       << expected_offset << endl;

	// on the first read of each loop, the first small_block samples
	// of data_small should equal those of data_large
	if (ismall == 0) {

	  unsigned char* bytes_small = data_small->get_rawptr();
	  unsigned char* bytes_large = data_large->get_rawptr();

	  uint64 nbyte = data_small->get_nbytes();

	  for (unsigned ibyte=0; ibyte < nbyte; ibyte++) {
	    if (bytes_small[ibyte] != bytes_large[ibyte])
	      cerr << "ERROR: data[" << ibyte << "]"
		" small=" << bytes_small[ibyte] << " !="
		" large=" << bytes_large[ibyte] << endl;
	  }
	}

      }
      block++;
      if (block == blocks)
	break;
    }

    if (verbose)
      cerr << "end of data file " << filenames[ifile] << endl;

  }
  catch (string& error) {
    cerr << error << endl;
  }

  return 0;
}

catch (Error& error) {
  cerr << "Error thrown: " << error << endl;
  return -1;
}

catch (Reference::invalid& error) {
  cerr << "Reference invalid exception thrown" << endl;
  return -1;
}

catch (string& error) {
  cerr << "exception thrown: " << error << endl;
  return -1;
}

catch (...) {
  cerr << "unknown exception thrown." << endl;
  return -1;
}

}
