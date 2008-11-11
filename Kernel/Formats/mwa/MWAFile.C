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

  info.set_start_time( MJD(52644.0) );
  string prefix = "mwa";

  info.set_telescope( "Greenbank" );
  info.set_source("J0534+2200");
  info.set_npol(2);
  info.set_ndim(1);
  info.set_nbit(32);
  info.set_type( Signal::Pulsar );
  info.set_mode("32-bit");
  info.set_state( Signal::Nyquist );
  info.set_machine( "MWA" );
  
  info.set_nchan(1);
  info.set_bandwidth(8.0);
  info.set_rate( fabs(2.0e6*info.get_bandwidth())/double(info.get_nchan()) );

  info.set_scale( 1.0 );
  info.set_swap( false );
  info.set_dc_centred( false );
  info.set_calfreq( 0.0 );
  info.set_dispersion_measure( 0.0 );

  {
    // COORDINATES are stored as RAJ and DECJ
    string raj = "05:34:00";
    string decj = "22:00:00";

    sky_coord s;
    s.setHMSDMS(raj.c_str(),decj.c_str());
    info.set_coordinates( s );
  }

  info.set_centre_frequency( 200.0 );

  {
    uint64 bits_per_sample = info.get_nbit() * info.get_nchan() * info.get_npol();
    
    uint64 data_bits = 8*filesize(filename);
    
    info.set_ndat( data_bits / bits_per_sample );
  }

  //  fprintf(stderr,"Going to do dump:\n\n\n");
  //info.print();

  fd = ::open(filename,O_RDONLY);

  if( fd < 0 )
    throw Error(FailedCall,"dsp::DumbLBAFile::open_file()",
		"Failed to open file '%s'",filename);

  header_bytes = 0;

}
