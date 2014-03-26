#include "filterbank.h"


/* define the varaiables declared extern in filterbank.h and header.h here */
/* input and output files and logfile (filterbank.monitor) */
FILE *input, *output, *logfile;
char  inpfile[80], outfile[80];

/* global variables describing the data */
double time_offset;

/* global variables describing the operating mode */
float start_time, final_time, clip_threshold;

int obits, sumifs, headerless, headerfile, swapout, invert_band;
int compute_spectra, do_vanvleck, hanning, hamming, zerolagdump;
int headeronly;
char ifstream[8];

/* global variables describing the data */
char rawdatafile[80], source_name[80];
int machine_id, telescope_id, data_type, nchans, nbits, nifs, scan_number,
  barycentric,pulsarcentric; /* these two added Aug 20, 2004 DRL */
double tstart,mjdobs,tsamp,fch1,foff,refdm,az_start,za_start,src_raj,src_dej;
double gal_l,gal_b,header_tobs,raw_fch1,raw_foff;
int nbeams, ibeam;
char isign;
/* added 20 December 2000    JMC */
double srcl,srcb;
double ast0, lst0;
long wapp_scan_number;
char project[8];
char culprits[24];
double analog_power[2];

/* added frequency table for use with non-contiguous data */
double frequency_table[4096]; /* note limited number of channels */
long int npuls; /* added for binary pulse profile format */



void filterbank_header(FILE *outptr) /* includefile */
{
  if (sigproc_verbose)
    fprintf (stderr, "sigproc::filterbank_header\n");

  int i,j;
  output=outptr;
  if (obits == -1) obits=nbits;
  /* go no further here if not interested in header parameters */

  if (headerless)
  {
    if (sigproc_verbose)
      fprintf (stderr, "sigproc::filterbank_header headerless - abort\n");
    return;
  }

  if (sigproc_verbose)
    fprintf (stderr, "sigproc::filterbank_header HEADER_START");

  send_string("HEADER_START");
  if (!strings_equal(inpfile,""))
  {
    send_string("rawdatafile");
    send_string(inpfile);
  }
  if (!strings_equal(source_name,""))
  {
    send_string("source_name");
    send_string(source_name);
  }
    send_int("machine_id",machine_id);
    send_int("telescope_id",telescope_id);
    send_coords(src_raj,src_dej,az_start,za_start);
    if (zerolagdump) {
      /* time series data DM=0.0 */
      send_int("data_type",2);
      refdm=0.0;
      send_double("refdm",refdm);
      send_int("nchans",1);
    } else {
      /* filterbank data */
      send_int("data_type",1);
      send_double("fch1",fch1);
      send_double("foff",foff);
      send_int("nchans",nchans);
    }
    /* beam info */
    send_int("nbeams",nbeams);
    send_int("ibeam",ibeam);
    /* number of bits per sample */
    send_int("nbits",obits);
    /* start time and sample interval */
    send_double("tstart",tstart+(double)start_time/86400.0);
    send_double("tsamp",tsamp);
    if (sumifs) {
      send_int("nifs",1);
    } else {
      j=0;
      for (i=1;i<=nifs;i++) if (ifstream[i-1]=='Y') j++;
      if (j==0) error_message("no valid IF streams selected!");
      send_int("nifs",j);
    }
    send_string("HEADER_END");
}
  

