#include "dsp/PuMaFile.h"

#include "Error.h"
#include "string_utils.h"

#include <libpuma.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>


dsp::PuMaFile::PuMaFile (const char* filename)
  : File ("PuMa")
{
  //! The PuMa header
  header = malloc (sizeof(Header_type));

  if (filename)
    open (filename);
}

dsp::PuMaFile::~PuMaFile ()
{
  free (header);
}

bool dsp::PuMaFile::is_valid (const char* filename) const
{ 
  return false;
}

void dsp::PuMaFile::open_file (const char* filename)
{
  if (verbose)
    cerr << "dsp::PuMaFile::open_file " << filename << endl;

  FILE* fptr = fopen (filename, "r");
  if (!fptr)
    throw Error (FailedSys, "dsp::PuMaFile::open",
		 "failed fopen(%s)", filename);
 
  Header_type* hdr = (Header_type*) hdr;

  // call the parsing routine from the puma library
  prheader (hdr, fptr);
  
  if (verbose)
    cerr << "dsp::PuMaFile::open_file Skipping adjustments" << endl;
  
  // skip over adjustments
  
  /* int ParBlkSize; Number of bytes in second (parameter) block */
  unsigned nadjust = hdr->gen.ParBlkSize / (sizeof(Adjustments));
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
  
void dsp::PuMaFile::set_info (const void* header)
{
  const Header_type* hdr = (const Header_type*) header;

  /* Boolean Raw;    In case of raw data */
  if (hdr->redn.Raw != true)
    throw Error (InvalidState, "dsp::PuMaFile::set_info",
		 "Data file does not contain raw data");

  /* Boolean Cluster[MAXFREQBANDS]; Is Data from this cluster in this file? */
  unsigned iband = MAXFREQBANDS;
  
  for (unsigned i=0; i < MAXFREQBANDS; i++)
    if (hdr->gen.Cluster[iband] == true) {
      if (iband != MAXFREQBANDS)
	throw Error (InvalidState, "parsePuMaHeader",
		     "More than one cluster in data file");
      iband = i;
    }
  
  if (iband == MAXFREQBANDS)
    throw Error (InvalidState, "parsePuMaHeader", "Cluster not set");
  
  /* Until further notice */
  info.set_basis (Signal::Linear);

  /* char ObsType[TXTLEN]; Research/test/calibration */
  cerr << "parsePuMaHeader type = " << hdr->obs.ObsType << endl;
  info.set_type (Signal::Pulsar);

  unsigned npol_observed = 0;
  bool invalid = false;
  
  /* int Nr; Mode of PuMa observation -1,0,1,2,4 */
  if (hdr->mode.Nr == 0) {
    
    info.set_state (Signal::Nyquist);
    
    /* Boolean Xout;  Mode -1,0; X pol output    Mode 1,2,4  : FALSE */
    if (hdr->mode.Xout)
      npol_observed ++;
    
    /* Boolean Yout;  Mode -1,0; Y pol output    Mode 1,2,4  : FALSE */
    if (hdr->mode.Yout)
      npol_observed ++;
    
  }
  else if (hdr->mode.Nr == 1) {
    
    /* Boolean Iout; Mode 1,2 ; I pol output    Mode -1,0,4 : FALSE */
    /* Boolean Qout; Mode 1,2 ; Q pol output    Mode -1,0,4 : FALSE */
    /* Boolean Uout; Mode 1,2 ; U pol output    Mode -1,0,4 : FALSE */
    /* Boolean Vout; Mode 1,2 ; V pol output    Mode -1,0,4 : FALSE */
    bool pol_vect = hdr->mode.Qout && hdr->mode.Uout && hdr->mode.Vout;
    bool pol_some = hdr->mode.Qout || hdr->mode.Uout || hdr->mode.Vout;
    
    if (hdr->mode.Iout && pol_vect) {
      info.set_state (Signal::Stokes);
      npol_observed = 4;
    }
    else if (hdr->mode.Iout && !pol_some) {
      info.set_state (Signal::Intensity);
      npol_observed = 1;
    }
    else
      invalid = true;
    
  }
  else
    invalid = true;
  
  if (invalid)
    throw Error (InvalidState, "parsePuMaHeader", "unknown Mode");

  if (info.get_state() == Signal::Nyquist)
    info.set_mode ("Mode 0");
  else
    info.set_mode ("Mode 1");


  info.set_npol (npol_observed);

  /* int NFreqInFile; Mode 0-4 ; 1,2,4,8,16,..FreqChans in this file.
     Normally one cluster per file is written. */
  info.set_nchan (hdr->mode.NFreqInFile);
  
  /* for now data is always real-valued */
  info.set_ndim (1);
  
  /* int BitsPerSamp; Number of output bits ; 1,2,4,8 */
  info.set_nbit (hdr->mode.BitsPerSamp);

  /* int DataBlkSize; Number of bytes in data block */
  info.set_ndat( info.get_nsamples (hdr->gen.DataBlkSize) );
  
  string Westerbork = "WESTERBORK";
    
  /* char Name[NAMELEN];  Name of the observatory  */
  if (hdr->obsy.Name != Westerbork)
    throw Error (InvalidState, "parsePuMaHeader",
		 "Observatory name != " + Westerbork);
  
  // Always Westerbork for now
  info.set_telescope_code ('i');
  
  /* char Pulsar[NAMELEN]; Using the Taylor (1993) convention.
     e.g. "PSR J0218+4232" */
  string pulsar = hdr->src.Pulsar;
  
  // strip off the "PSR"
  stringtok (&pulsar, " ");
  cerr << "parsePuMaHeader source='" << pulsar << "'" << endl;
  
  info.set_source (pulsar);
  
  sky_coord position;
  
  /* double RA;  RA of the target (in radians)  */
  position.ra().setRadians( hdr->src.RA );
  /* double Dec; Dec of the target (in radians) */
  position.dec().setRadians( hdr->src.Dec );
  
  info.set_coordinates( position );
  
  /* double SkyMidFreq; Mid sky frequency of band (in MHz) */
  info.set_centre_frequency( hdr->WSRT.Band[iband].SkyMidFreq );
  
  double sign = 1.0;
  /* Boolean NonFlip; Is band flipped (reverse freq order within band)? */
  if (hdr->WSRT.Band[iband].NonFlip == false)
    sign = -1.0;
  
  /* double Width; Width of the band (in MHz): 2.5, 5.0 or 10.0 */
  info.set_bandwidth( sign * hdr->WSRT.Band[iband].Width );

  /* int DataMJD;     MJD of first sample of this data block */
  /* double DataTime; Time of first sample (fraction of day) */
  info.set_start_time( MJD( hdr->gen.DataMJD, hdr->gen.DataTime ) );
  
  /* int Tsamp; Mode 0-4 ; output sample interval in nano sec */
  double sampling_interval = 1e-9 * double(hdr->mode.Tsamp);
  info.set_rate (1.0/sampling_interval);
  
  info.set_scale (1.0);
  
  info.set_swap (false);
  
  info.set_dc_centred (false);
  
  /* char ScanNum[NAMELEN]; FileSeriesNumber(FF)+Cluster(C). */
  info.set_identifier (hdr->gen.ScanNum);
  
  /* char Platform[NAMELEN]; Name of machine: "PuMa" or "FFB" */
  info.set_machine (hdr->gen.Platform);
  
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
