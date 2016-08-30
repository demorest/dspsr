/***************************************************************************
 *
 *   Copyright (C) 2006 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include "dsp/MWAFile.h"
#include "dirutil.h"

#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>

using namespace std;

//! Construct and open file
dsp::MWAFile::MWAFile (const char* filename)
  : File("MWA")
{
}

//! Virtual destructor
dsp::MWAFile::~MWAFile()
{
}

//! Returns true if filename appears to name a valid LBA file
bool
dsp::MWAFile::is_valid (const char* filename) const
{
  if( verbose )
    fprintf(stderr,"Entered dsp::MWAFile::is_valid(%s)\n",filename);

  if( !file_exists(filename) ){
    throw Error(InvalidState,"dsp::MWAFile::is_valid()",
		"File '%s' didn't exist!",
		filename);
    return false;
  }

  if (strstr (filename, ".mwa"))
    return true;

  return false;
}

//! Open the file
void
dsp::MWAFile::open_file (const char* filename)
{
  if( verbose )
    fprintf(stderr,"Entered dsp::MWAFile::open_file (%s)\n",filename);

  if( !is_valid(filename) )
    throw Error(InvalidState,"dsp::MWAFile::open_file()",
		"filename '%s' was not a valid LBA file",
		filename);

  get_info()->set_start_time( MJD(52644.0) );
  string prefix = "mwa";

  get_info()->set_telescope( "Greenbank" );
  get_info()->set_source("J0534+2200");
  get_info()->set_npol(2);
  get_info()->set_ndim(1);
  get_info()->set_nbit(32);
  get_info()->set_type( Signal::Pulsar );
  get_info()->set_mode("32-bit");
  get_info()->set_state( Signal::Nyquist );
  get_info()->set_machine( "MWA" );
  
  get_info()->set_nchan(1);
  get_info()->set_bandwidth(8.0);
  get_info()->set_rate( fabs(2.0e6*get_info()->get_bandwidth())/double(get_info()->get_nchan()) );

  get_info()->set_scale( 1.0 );
  get_info()->set_swap( false );
  get_info()->set_dc_centred( false );
  get_info()->set_calfreq( 0.0 );
  get_info()->set_dispersion_measure( 0.0 );

  {
    // COORDINATES are stored as RAJ and DECJ
    string raj = "05:34:00";
    string decj = "22:00:00";

    sky_coord s;
    s.setHMSDMS(raj.c_str(),decj.c_str());
    get_info()->set_coordinates( s );
  }

  get_info()->set_centre_frequency( 200.0 );

  {
    uint64_t bits_per_sample = get_info()->get_nbit() * get_info()->get_nchan() * get_info()->get_npol();
    
    uint64_t data_bits = 8*filesize(filename);
    
    get_info()->set_ndat( data_bits / bits_per_sample );
  }

  //  fprintf(stderr,"Going to do dump:\n\n\n");
  //get_info()->print();

  fd = ::open(filename,O_RDONLY);

  if( fd < 0 )
    throw Error(FailedCall,"dsp::DumbLBAFile::open_file()",
		"Failed to open file '%s'",filename);

  header_bytes = 0;

}
