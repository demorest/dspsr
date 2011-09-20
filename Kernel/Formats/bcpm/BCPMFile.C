/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BCPMFile.h"

#include "dsp/bpphdr.h"
#include "machine_endian.h"

#include "FilePtr.h"
#include "tostring.h"
#include "dirutil.h"

#include <fstream>
#include <stdlib.h>
#include <string.h>

using namespace std;

//#include "f772c.h"
#ifndef F772C
#define F772C(x) x##_
#endif

extern "C" {
  void F772C(indexx)(int* n,float* arrin, int* indx);
}

//! Construct and open file
dsp::BCPMFile::BCPMFile (const char* filename) : File ("BCPM")
{
  if (filename)
    open (filename);
}

string read_line (const char* filename)
{
  ifstream input (filename);
  if (!input)
    throw Error (FileNotFound, "read_line",
                "Failed to open file '%s'", filename);

  string line;
  getline(input,line);
  return line;
}

//! Returns true if filename appears to name a valid BCPM file
bool dsp::BCPMFile::is_valid (const char* filename) const
{
  FilePtr ptr = fopen (filename, "r");
  if (!ptr)
    return false;

  char first[16];
  fgets (first, 16, ptr);
  first[10] = '\0';

  char expect[16] = "NBPPSEARCH";

  if (strcmp (first, expect) == 0)
    return true;

  if (verbose)
    cerr << "dsp::BCPMFile::is_valid first 10 characters '" << first << "'"
            " != '" << expect << "'" << endl;

  return false;
}

//! Switches the endianess of relevant variables, if need be
void from_big_endian (BPP_SEARCH_HEADER& bpp_search)
{
  FromBigEndian(bpp_search.samp_rate);
  FromBigEndian(bpp_search.bandwidth);
  FromBigEndian(bpp_search.rf_lo);
  FromBigEndian(bpp_search.bit_mode);
  FromBigEndian(bpp_search.num_chans);

  FromBigEndian(bpp_search.scan_file_number);
  FromBigEndian(bpp_search.file_size);
  FromBigEndian(bpp_search.ll_file_offset);

  FromBigEndian(bpp_search.mb_start_address);
  FromBigEndian(bpp_search.mb_end_address);
  FromBigEndian(bpp_search.mb_start_board);
  FromBigEndian(bpp_search.mb_end_board);
  for (unsigned i=0;i<MAXNUMCB;i++) 
    FromBigEndian(bpp_search.cb_id[i]);
  for (unsigned i=0;i<MAX_NUM_LO_BOARDS;i++) 
    FromBigEndian(bpp_search.aib_los[i]);
  for (unsigned i=0;i<FB_CHAN_PER_BRD;i++)
    FromBigEndian(bpp_search.dfb_sram_freqs[i]);
  FromBigEndian(bpp_search.ra_1950);
  FromBigEndian(bpp_search.dec_1950);
  FromBigEndian(bpp_search.cb_sum_polarizations);
  FromBigEndian(bpp_search.BACKEND_TYPE);
}

//! Open the file
void dsp::BCPMFile::open_file (const char* filename)
{
  if (verbose)
    cerr << "dsp::BCPMFile::open_file(" << filename << ")" << endl;

  FILE* fptr = fopen (filename,"r");
  if (!fptr)
    throw Error (FailedSys, "dsp::BCPMFile::open_file", "fopen(%s)", filename);

  BPP_SEARCH_HEADER bpp_search;

  fread(&bpp_search,sizeof(BPP_SEARCH_HEADER),1,fptr);

  fclose(fptr);

  from_big_endian (bpp_search);

  if (bpp_search.BACKEND_TYPE != 0 && bpp_search.BACKEND_TYPE != 1 && bpp_search.BACKEND_TYPE != 4)
    throw Error(InvalidParam,"dsp::BCPMFile::open_file()",
		"This header doesn't fulfill the requirement from sigproc that the bpp_search.BACKEND_TYPE is 0 or 1 or 4!  (It is %d)",
		bpp_search.BACKEND_TYPE);

  if( !bpp_search.cb_sum_polarizations )
    throw Error(InvalidParam,"dsp::BCPMFile::open_file()",
		"bpp_search.cb_sum_polarizations is false");

  get_info()->set_telescope( "Tidbinbilla" );
  get_info()->set_machine( "BCPM" );

  get_info()->set_nchan( unsigned(bpp_search.num_chans) );
  get_info()->set_ndim( 1 );
  get_info()->set_npol( 1 );
  get_info()->set_nbit( unsigned(bpp_search.bit_mode) );

  get_info()->set_scale( 1.0 );
  get_info()->set_swap( false );
  get_info()->set_mode("SEARCH");
  get_info()->set_dispersion_measure( 0.0 );
  get_info()->set_dc_centred( false );
  get_info()->set_type( Signal::Pulsar );
  get_info()->set_state( Signal::Intensity );
  get_info()->set_basis( Signal::Circular );
  get_info()->set_rate( 1.0e6/bpp_search.samp_rate );
  get_info()->set_source( bpp_search.target_name );

  try {
    // parse search date dayno:year
    char cdate[4096];
    sprintf(cdate,"%s",bpp_search.date);
    string dayno = strtok(cdate,":");
    string year = strtok(NULL,":");
    
    // parse UT start time hh:mm:ss
    string start_time = bpp_search.start_time;
    string utcstring = year + "-" + dayno + "-" + start_time;

    if (verbose)
    {
      cerr << "dayno=" << dayno << endl;    
      cerr << "year=" << year << endl;
      cerr << "start_time=" << start_time << endl;
      cerr << "scan_file_number=" << bpp_search.scan_file_number << endl;
      cerr << "file_size=" << bpp_search.file_size << endl;
      cerr << "ll_file_offset=" << bpp_search.ll_file_offset << endl;
    }

    utc_t my_utc;
    str2utc(&my_utc, utcstring.c_str());

    // utc_t.tm_yday starts counting at 1. dayno starts counting at 0.
    MJD mjd = MJD(my_utc) + MJD(1.0);

    /*
      Willem van Straten - 21 September 2011

      The following attempt to compute the time offset of the start of 
      the current file from the start of the observation will typically
      fail on the last file of the scan, which will typically have a
      file_size that is less than all of the preceding files.

      Unfortunately, ll_file_offset == 0 in all of the files seen to date.
    */
    uint64_t file_offset = bpp_search.scan_file_number - 1;
    uint64_t byte_offset = bpp_search.file_size * file_offset;

    mjd += get_info()->get_nsamples(byte_offset) / get_info()->get_rate();
    get_info()->set_start_time( mjd );

    char id[15];
    utc2str(id, my_utc, "yyyydddhhmmss");
    get_info()->set_identifier( id );
  }
  catch (Error& error)
  {
    error << "\n\t" "while parsing start time";
    throw error += "dsp::BCPMFile::open_file";
  }

  {
    string rab = tostring(bpp_search.ra_1950);
    string decb = tostring(bpp_search.dec_1950);
      
    sky_coord coords;
    coords.setHMSDMS(rab.c_str(),decb.c_str());

    get_info()->set_coordinates( coords );
  }

  {
    uint64_t file_bytes = filesize(filename);
    uint64_t data_bytes = file_bytes - BPP_HEADER_SIZE;
    get_info()->set_ndat( get_info()->get_nsamples(data_bytes) );
  }

  double centre_frequency = 0.0;
  double ch_bw = 0.0;

  bpp_chans( chtab,
	     bpp_search.bandwidth,bpp_search.mb_start_address,
	     bpp_search.mb_end_address,bpp_search.mb_start_board,
	     bpp_search.mb_end_board,bpp_search.cb_id,
	     bpp_search.aib_los,bpp_search.dfb_sram_freqs,
	     bpp_search.rf_lo,
	     centre_frequency, ch_bw );  

  get_info()->set_centre_frequency( centre_frequency );
  get_info()->set_bandwidth( fabs(ch_bw) * get_info()->get_nchan() );

  if (verbose && sizeof(BPP_SEARCH_HEADER) != BPP_HEADER_SIZE)
  {
    cerr << "dsp::BCPMFile::open_file WARNING "
      "BPP_HEADER_SIZE=" << BPP_HEADER_SIZE << " != "
      "sizeof(BPP_SEARCH_HEADER)=" << sizeof(BPP_SEARCH_HEADER) <<
      " (using BPP_HEADER_SIZE)"
	 << endl;
  }

  // Note that BPP_HEADER_SIZE is 8 bytes bigger than sizeof(BPP_SEARCH_HEADER) so I'm not sure if this is right
  header_bytes = BPP_HEADER_SIZE;

  open_fd (filename);

  // cannot load less than a byte. set the time sample resolution accordingly
  unsigned bits_per_byte = 8;
  resolution = bits_per_byte / get_info()->get_nbit();
  if (resolution == 0)
    resolution = 1;
}

//! Pulled out of sigproc
void dsp::BCPMFile::bpp_chans(vector<int>& table,
			      double bw, int mb_start_addr, int mb_end_addr, int mb_start_brd, int mb_end_brd, int *cb_id, double *aib_los, float *dfb_sram_freqs, double rf_lo,
			      double& centre_frequency,
			      double& channel_bandwidth)
{
  static int dfb_chan_lookup[MAXREGS][NIBPERREG] = {
    {4, 0, 4, 0},
    {5, 1, 5, 1},
    {6, 2, 6, 2}, 
    {7, 3, 7, 3},
    {4, 0, 4, 0},
    {5, 1, 5, 1},
    {6, 2, 6, 2},
    {7, 3, 7, 3}
  };
  
  /* This takes care of byte swap in outreg_b */
  static float sideband_lookup[MAXREGS][NIBPERREG] = {
    {-1.0, -1.0, +1.0, +1.0},
    {-1.0, -1.0, +1.0, +1.0},
    {-1.0, -1.0, +1.0, +1.0},
    {-1.0, -1.0, +1.0, +1.0},
    {+1.0, +1.0, -1.0, -1.0},
    {+1.0, +1.0, -1.0, -1.0},
    {+1.0, +1.0, -1.0, -1.0},
    {+1.0, +1.0, -1.0, -1.0}
  };
  
  int i, n=0, dfb_chan, logical_brd, nibble;
  double  f_aib, u_or_l, f_sram;
  
  int nchans = (mb_end_addr/2-mb_start_addr/2+1)*(mb_end_brd-mb_start_brd+1)*4;
  float* fmhz   = (float *) malloc(nchans*sizeof(float));
  table.resize(nchans);
  int* nridx  = (int*) malloc(nchans*sizeof(int));

  double rf_lo_mhz = rf_lo/1.e6;

  if (-1.e6<rf_lo && rf_lo<1.e6) 
    rf_lo_mhz = rf_lo;

  /* 
     Loop over (16-bit) regs per board. divide by 2's are to make them 
     word addresses instead of byte addresses so we can index with them.
     Normal modes will be regid = 0..3, 0..7, or 4..7 
  */

  for(int regid=mb_start_addr/2; regid<=mb_end_addr/2; regid++){
    /* Loop over each board */
    for (int bid=mb_start_brd;bid<=mb_end_brd;bid++) {
      /* Now find which LOGICAL CB we are reading */
      logical_brd = -1;
      for (i=0; i<MAXNUMCB; i++) {
        if (bid == cb_id[i]) {
	  logical_brd = i;
	  break;
        }
      }
      if (logical_brd == -1)
	throw Error(InvalidState,"dsp::BCPMFile::bpp_chans()",
		    "bpp_chan - logical_brd not found");

      /* Assumes cabling so that LO0 feeds MF0,1 which feeds leftmost CB! */
      f_aib = aib_los[logical_brd];
      /* Loop over 4 nibbles per reg */
      for (nibble=0; nibble<4; nibble++) {
        dfb_chan = dfb_chan_lookup[regid][nibble];
        u_or_l = sideband_lookup[regid][nibble];
        f_sram = dfb_sram_freqs[dfb_chan];
        double fc = f_aib + f_sram + u_or_l * bw/4.0;
	if (rf_lo_mhz<1.e4) /* below 10 GHz LSB; above 10 GHz USB */
	  fmhz[n++]=rf_lo_mhz+800-fc/1.0e6;
	else
	  fmhz[n++]=rf_lo_mhz+fc/1.0e6;
      }
    }
  }

  /* produce lookup table which gives channels in order of descending freq */
  int ninetysix = 96;
  F772C(indexx)(&ninetysix,fmhz,nridx);
  if (nchans==192)
  {
    F772C(indexx)(&ninetysix,fmhz+96,nridx+96);
    for (i=96; i<192; i++) 
      nridx[i] += 96;
  }

  for (i=0;i<nchans;i++) 
    table[i]=nridx[i]-1;    

  centre_frequency = 0.5*(fmhz[table[0]]+fmhz[table[95]]); 

  channel_bandwidth = fmhz[table[1]] - fmhz[table[0]];

  //  for( unsigned i=0; i<96; i++)
  //fprintf(stderr, "bpp_chans: fmhz[%d]=%f fmhz[table[%d]]=fmhz[%d]=%f\n",
  //    i,fmhz[i],i,table[i],fmhz[table[i]]);

  free(fmhz);
  free(nridx);
}

//! Pads gaps in data
int64_t dsp::BCPMFile::pad_bytes(unsigned char* buffer, int64_t bytes){
  if( get_info()->get_nbit() < 8 && bytes*8 % int64_t(get_info()->get_nbit()) != 0 )
    throw Error(InvalidState,"dsp::BCPMFile::pad_bytes()",
		"number of input bits is not divisible by nbit=%d",
		get_info()->get_nbit());

  register const unsigned char val = 0;
  for( int64_t i=0; i<bytes; ++i)
    buffer[i] = val;
  
  return bytes;
}
