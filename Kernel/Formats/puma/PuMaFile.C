#include "PuMaFile.h"
#include "PuMaObservation.h"

#include "libpuma.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

void dsp::PuMaFile::open_file (const char* filename)
{
  if (verbose)
    cerr << "dsp::PuMaFile::open " << filename << endl;

  FILE* fptr = fopen (filename, "r");
  if (!fptr)
    throw Error (FailedSys, "dsp::PuMaFile::open",
		 "failed fopen(%s)", filename);
  
  // call the parsing routine from the puma library
  prheader (&hdr, fptr);
  
  if (verbose)
    cerr << "dsp::PuMaFile::::openFile Skipping adjustments" << endl;
  
  // skip over adjustments
  
  /* int ParBlkSize; Number of bytes in second (parameter) block */
  unsigned nadjust = hdr.gen.ParBlkSize / (sizeof(Adjustments));
  fseek(fptr, nadjust*sizeof(Adjustments), SEEK_CUR);
  
  header_bytes = ftell (fptr);
  
  fclose (fptr);
  
  set_info (hdr);
  
  fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::PuMaFile::open", 
		 "failed open(%s)", filename);

  if (verbose)
    cerr << "dsp::PuMaFile::open exit" << endl;
}
  
void dsp::PuMaFile::set_info (const Header_type& hdr)
{
  /* Boolean Raw;    In case of raw data */
  if (hdr.redn.Raw != true)
    throw Error (InvalidState, "dsp::PuMaFile::set_info",
		 "Data file does not contain raw data");

  PuMa::set_observation (info, hdr);

  /* int DataMJD;     MJD of first sample of this data block */
  /* double DataTime; Time of first sample (fraction of day) */
  info.set_start_time( MJD( hdr.gen.DataMJD, hdr.gen.DataTime ) );
  
  /* int Tsamp; Mode 0-4 ; output sample interval in nano sec */
  double sampling_interval = 1e-9 * double(hdr.mode.Tsamp);
  info.set_rate (1.0/sampling_interval);
  
  info.set_scale (1.0);
  
  info.set_swap (false);
  
  info.set_dc_centred (false);
  
  /* char ScanNum[NAMELEN]; FileSeriesNumber(FF)+Cluster(C). */
  info.set_identifier (hdr.gen.ScanNum);
  
  /* char Platform[NAMELEN]; Name of machine: "PuMa" or "FFB" */
  info.set_machine (hdr.gen.Platform);
  
  info.set_dispersion_measure (0);
  info.set_between_channel_dm (0);
    
}


  

#if FUTURE_WORK

//! Produce a string of the form ".00001.1.puma"
string dsp::PuMaFile::make_fname_extension (int itimediv, int iband)
{
  return stringprintf (".%05d.%1d.puma", itimediv, iband);
}

FILE* dsp::PuMaFile::::openFile (int itimediv, int iband, Header_type *hdrptr)
{
  // construct file name and open file
  string fname = fprefix + make_fname_extension (itimediv, iband);

  if (verbose)
    cerr << "dsp::PuMaFile::::openFile '" << fname << "'" << endl;

  FILE *f = fopen(fname, "r");
  // read in header
  Header_type  hdr;
  if (hdrptr==NULL)
    hdrptr = &hdr;

  prheader (hdrptr, f);

  if (verbose)
    cerr << "dsp::PuMaFile::::openFile Skipping adjustments" << endl;

  // skip over adjustments
  int nadjust = hdrptr->gen.ParBlkSize / (sizeof(Adjustments));
  fseek(f, nadjust*sizeof(Adjustments), SEEK_CUR);

  return f;
}

// function to open all the files for a given time division
void dsp::PuMaFile::openFiles(int timediv)
{
  int i;

  for (i=0; i < nbands; i++)
  {
    band_fptr[i] = openFile(timediv, ibandstart+i);
    band_samplestart_offset[i] = ftell(band_fptr[i]);
  }
  current_offset = offset_bydiv[timediv];
  current_div = timediv;
}

void dsp::PuMaFile::closeFiles()
{
  int i;

  for (i=0; i < nbands; i++)
    fclose (band_fptr[i]);
}

#endif
