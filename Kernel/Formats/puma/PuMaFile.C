#include "dsp/PuMaFile.h"

#include "Error.h"
#include "string_utils.h"

#if !defined(MALIGN_DOUBLE)
#define NO_MALIGN_DOUBLE
#endif

#include "libpuma.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

extern "C" void pumadump (const Header_type *hdr, FILE* fptr, Boolean verb);

dsp::PuMaFile::PuMaFile (const char* filename)
  : File ("PuMa")
{
  //! The PuMa header
  unsigned header_size = sizeof (Header_type);

  if (header_size != 4504) {
    Error::verbose = true;
    throw Error (InvalidState, "dsp::PuMaFile", 
                 "sizeof(Header_type)=%d != 4504", header_size);
  }

  header = malloc (sizeof(Header_type));

  if (filename)
    open (filename,0);
}

dsp::PuMaFile::~PuMaFile ()
{
  free (header);
}

bool dsp::PuMaFile::is_valid (const char* filename,int) const
{ 
  return true; // false;
}

void dsp::PuMaFile::open_file (const char* filename)
{
  if (verbose)
    cerr << "dsp::PuMaFile::open_file " << filename << endl;

  FILE* fptr = fopen (filename, "r");
  if (!fptr)
    throw Error (FailedSys, "dsp::PuMaFile::open",
		 "failed fopen(%s)", filename);
 
  Header_type* hdr = (Header_type*) header;

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

  if (verbose)
    cerr << "dsp::PuMaFile::open_file parse header" << endl;

  parse (hdr);

  if (verbose)
    cerr << "dsp::PuMaFile::open_file call open(" << filename << ")" << endl;

  fd = ::open (filename, O_RDONLY);
  if (fd < 0)
    throw Error (FailedSys, "dsp::PuMaFile::open", 
		 "failed open(%s)", filename);

  if (verbose)
    cerr << "dsp::PuMaFile::open exit" << endl;
}
  
void dsp::PuMaFile::parse (const void* header)
{
  const Header_type* hdr = (const Header_type*) header;

  if (verbose)
    pumadump (hdr, stderr, verbose);

  /* Boolean Raw;    In case of raw data */
  if (hdr->redn.Raw != true)
    throw Error (InvalidState, "dsp::PuMaFile::parse",
		 "Data file does not contain raw data");

  /* Boolean Cluster[MAXFREQBANDS]; Is Data from this cluster in this file? */
  unsigned iband = MAXFREQBANDS;
  
  for (unsigned i=0; i < MAXFREQBANDS; i++)
    if (hdr->gen.Cluster[i]) {
      if (iband != MAXFREQBANDS)
	throw Error (InvalidState, "dsp::PuMaFile::parse",
		     "More than one cluster in data file (%d and %d)",
		     iband, i);
      iband = i;
    }
  
  if (iband == MAXFREQBANDS)
    throw Error (InvalidState, "dsp::PuMaFile::parse", "Cluster not set");

  if (verbose)
    cerr << "dsp::PuMaFile::parse Cluster " << iband << " in this file" << endl;

  /* Until further notice */
  info.set_basis (Signal::Linear);

  /* char ObsType[TXTLEN]; Research/test/calibration */
  // cerr << "dsp::PuMaFile::parse type = " << hdr->obs.ObsType << endl;
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
    throw Error (InvalidState, "dsp::PuMaFile::parse", "unknown Mode");

  if (info.get_state() == Signal::Nyquist)
    info.set_mode( string("Mode 0") );
  else
    info.set_mode( string("Mode 1") );

  info.set_npol (npol_observed);

  /* int NFreqInFile; Mode 0-4 ; 1,2,4,8,16,..FreqChans in this file.
     Normally one cluster per file is written. */
  info.set_nchan (hdr->mode.NFreqInFile);
  
  /* for now data is always real-valued */
  info.set_ndim (1);

  /* int BitsPerSamp; Number of output bits ; 1,2,4,8 */
  info.set_nbit (hdr->mode.BitsPerSamp);

  if (verbose)
    cerr << "dsp::PuMaFile::parse " << info.get_nbyte() << " bytes/sample" 
         << endl;

#if 0
  string Westerbork = "WESTERBORK";
    
  /* char Name[NAMELEN];  Name of the observatory  */
  if (hdr->obsy.Name != Westerbork)
    Error (InvalidState, "dsp::PuMaFile::parse",
		 "Observatory name='%s' != '%s'",
		 hdr->obsy.Name, Westerbork.c_str());
#endif

  // Always Westerbork for now
  info.set_telescope_code ('i');

  /* char Pulsar[NAMELEN]; Using the Taylor (1993) convention.
     e.g. "PSR J0218+4232" */
  string pulsar = hdr->src.Pulsar;
  
  // strip off the "PSR"
  string prefix = stringtok (&pulsar, " _");

  if (prefix != "PSR")
    pulsar = prefix;

  if (verbose)
    cerr << "dsp::PuMaFile::parse source='" << pulsar << "'" << endl;
  
  info.set_source (pulsar);

  sky_coord position;
  
  /* double RA;  RA of the target (in radians)  */
  position.ra().setRadians( hdr->src.RA );
  /* double Dec; Dec of the target (in radians) */
  position.dec().setRadians( hdr->src.Dec );
  
  info.set_coordinates( position );
  
  double sign = 1.0;
  /* Boolean NonFlip; Is band flipped (reverse freq order within band)? */
  if (hdr->WSRT.Band[iband].NonFlip == false)
    sign = -1.0;
  
  int FIR_factor = hdr->mode.FIRFactor;
  cerr << "fir factor=" << FIR_factor << endl;

  /* double Width; Width of the band (in MHz): 2.5, 5.0 or 10.0 */
  info.set_bandwidth( sign * hdr->WSRT.Band[iband].Width / FIR_factor );

  /* double SkyMidFreq; Mid sky frequency of band (in MHz) */
  info.set_centre_frequency( hdr->WSRT.Band[iband].SkyMidFreq 
                             - 0.5 * (FIR_factor-1) * info.get_bandwidth() );

  /* int StMJD;       MJD at start of observation */
  /* int StTime;      Starttime (s after midnight, multiple of 10 s) */
  info.set_start_time( MJD( hdr->obs.StMJD, int(hdr->obs.StTime), 0.0 ) );

  if (verbose)
    cerr << "dsp::PuMaFile::parse start MJD=" << hdr->obs.StMJD
         << "d+" << hdr->obs.StTime << "s=" << info.get_start_time() << endl;

  /* int Tsamp; Mode 0-4 ; output sample interval in nano sec */
  double sampling_interval = 1e-9 * double(hdr->mode.Tsamp);
  info.set_rate (1.0/sampling_interval);

  /* int DataBlkSize; Number of bytes in data block */
  info.set_ndat( info.get_nsamples (hdr->gen.DataBlkSize) );

  /* The start time of the observation must be offset by the file number */
  /* int FileNum;     Which file out of NFiles is this - not reliable
                      as it counts over all bands. */

  uint64 filenum = hdr->gen.FileNum;
  int scanned = 0;

  char* filenum_str = strchr (hdr->gen.ThisFileName, '.');
  if (filenum_str)
    scanned = sscanf (filenum_str+1, UI64, &filenum);

  if (scanned != 1)
    throw Error (InvalidParam, "dsp::PuMaFile::parse",
                 "filename=%s not in recognized form", hdr->gen.ThisFileName);

  uint64 offset_samples = filenum * info.get_ndat();

  if (verbose)
    cerr << "dsp::PuMaFile::parse sampling rate=" << info.get_rate()
         << " ndat=" << info.get_ndat() << "\n\tfile=" << filenum
         << " offset=" << offset_samples << "samples = " 
         << offset_samples/info.get_rate() << "seconds" << endl;

  info.change_start_time (offset_samples);

  info.set_scale (1.0);
  
  info.set_swap (false);
  
  info.set_dc_centred (false);

  /* char ScanNum[NAMELEN]; FileSeriesNumber(FF)+Cluster(C). */
  info.set_identifier (hdr->gen.ScanNum);
  
  /* char Platform[NAMELEN]; Name of machine: "PuMa" or "FFB" */
  info.set_machine ("PuMa"); // (hdr->gen.Platform);
  
  info.set_dispersion_measure (0);
  info.set_between_channel_dm (0);

  // cannot load less than a byte. set the time sample resolution accordingly
  unsigned bits_per_byte = 8;
  resolution = bits_per_byte / info.get_nbit();
  if (resolution == 0)
    resolution = 1;

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
