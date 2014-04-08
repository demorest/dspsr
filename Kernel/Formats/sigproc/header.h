/* global variables describing the data */
extern char rawdatafile[80], source_name[80];
extern int machine_id, telescope_id, data_type, nchans, nbits, nifs, scan_number,
  barycentric,pulsarcentric; /* these two added Aug 20, 2004 DRL */
extern double tstart,mjdobs,tsamp,fch1,foff,refdm,az_start,za_start,src_raj,src_dej;
extern double gal_l,gal_b,header_tobs,raw_fch1,raw_foff;
extern int nbeams, ibeam;
extern char isign;
/* added 20 December 2000    JMC */
extern double srcl,srcb;
extern double ast0, lst0;
extern long wapp_scan_number;
extern char project[8];
extern char culprits[24];
extern double analog_power[2];

/* added frequency table for use with non-contiguous data */
extern double frequency_table[4096]; /* note limited number of channels */
extern long int npuls; /* added for binary pulse profile format */

// define the signedness for the 8-bit data type
#define OSIGN 1
#define SIGNED OSIGN < 0

