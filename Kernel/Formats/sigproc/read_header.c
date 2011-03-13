/* read_header.c - general handling routines for SIGPROC headers */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "header.h"
#include "sigproc.h"

char sigproc_verbose = 0;

int nbins;
double period;
/* read a string from the input which looks like nchars-char[1-nchars] */
void get_string(FILE *inputfile, int *nbytes, char string[]) /* includefile */
{
  int nchar;
  strcpy(string,"ERROR");
  fread(&nchar, sizeof(int), 1, inputfile);
  if (feof(inputfile)) exit(0);
  if (nchar>80 || nchar<1) return;
  *nbytes=sizeof(int);
  fread(string, nchar, 1, inputfile);
  string[nchar]='\0';
  *nbytes+=nchar;
}

/* attempt to read in the general header info from a pulsar data file */
int read_header (FILE *inputfile) /* includefile */
{
  char string[80], message[80];
  int itmp,nbytes,expecting_rawdatafile=0,expecting_source_name=0; 
  int expecting_frequency_table=0,channel_index=0;


  /* try to read in the first line of the header */
  get_string(inputfile,&nbytes,string);

  if (!strings_equal(string,"HEADER_START"))
  {
    if (sigproc_verbose)
      fprintf (stderr, "sigproc::read_header - no HEADER_START\n");

    /* the data file is not in standard format, rewind and return */
    rewind(inputfile);
    return 0;
  }

  /* loop over and read remaining header lines until HEADER_END reached */
  while (1) {
    get_string(inputfile,&nbytes,string);
    if (strings_equal(string,"HEADER_END")) break;
    if (strings_equal(string,"rawdatafile")) {
      expecting_rawdatafile=1;
    } else if (strings_equal(string,"source_name")) {
      expecting_source_name=1;
    } else if (strings_equal(string,"FREQUENCY_START")) {
      expecting_frequency_table=1;
      channel_index=0;
    } else if (strings_equal(string,"FREQUENCY_END")) {
      expecting_frequency_table=0;
    } else if (strings_equal(string,"az_start")) {
      fread(&az_start,sizeof(az_start),1,inputfile);
    } else if (strings_equal(string,"za_start")) {
      fread(&za_start,sizeof(za_start),1,inputfile);
    } else if (strings_equal(string,"src_raj")) {
      fread(&src_raj,sizeof(src_raj),1,inputfile);
    } else if (strings_equal(string,"src_dej")) {
      fread(&src_dej,sizeof(src_dej),1,inputfile);
    } else if (strings_equal(string,"tstart")) {
      fread(&tstart,sizeof(tstart),1,inputfile);
    } else if (strings_equal(string,"tsamp")) {
      fread(&tsamp,sizeof(tsamp),1,inputfile);
    } else if (strings_equal(string,"period")) {
      fread(&period,sizeof(period),1,inputfile);
    } else if (strings_equal(string,"fch1")) {
      fread(&fch1,sizeof(fch1),1,inputfile);
    } else if (strings_equal(string,"fchannel")) {
      fread(&frequency_table[channel_index++],sizeof(double),1,inputfile);
      fch1=foff=0.0; /* set to 0.0 to signify that a table is in use */
    } else if (strings_equal(string,"foff")) {
      fread(&foff,sizeof(foff),1,inputfile);
    } else if (strings_equal(string,"nchans")) {
      fread(&nchans,sizeof(nchans),1,inputfile);
    } else if (strings_equal(string,"telescope_id")) {
      fread(&telescope_id,sizeof(telescope_id),1,inputfile);
    } else if (strings_equal(string,"machine_id")) {
      fread(&machine_id,sizeof(machine_id),1,inputfile);
    } else if (strings_equal(string,"data_type")) {
      fread(&data_type,sizeof(data_type),1,inputfile);
    } else if (strings_equal(string,"ibeam")) {
      fread(&ibeam,sizeof(ibeam),1,inputfile);
    } else if (strings_equal(string,"nbeams")) {
      fread(&nbeams,sizeof(nbeams),1,inputfile);
    } else if (strings_equal(string,"nbits")) {
      fread(&nbits,sizeof(nbits),1,inputfile);
    } else if (strings_equal(string,"barycentric")) {
      fread(&barycentric,sizeof(barycentric),1,inputfile);
    } else if (strings_equal(string,"pulsarcentric")) {
      fread(&pulsarcentric,sizeof(pulsarcentric),1,inputfile);
    } else if (strings_equal(string,"nbins")) {
      fread(&nbins,sizeof(nbins),1,inputfile);
    } else if (strings_equal(string,"nsamples")) {
      /* read this one only for backwards compatibility */
      fread(&itmp,sizeof(itmp),1,inputfile);
    } else if (strings_equal(string,"nifs")) {
      fread(&nifs,sizeof(nifs),1,inputfile);
    } else if (strings_equal(string,"npuls")) {
      fread(&npuls,sizeof(npuls),1,inputfile);
    } else if (strings_equal(string,"refdm")) {
      fread(&refdm,sizeof(refdm),1,inputfile);
    } else if (strings_equal(string,"signed")) {
      fread(&isign,sizeof(isign),1,inputfile);
    } else if (expecting_rawdatafile) {
      strcpy(rawdatafile,string);
      expecting_rawdatafile=0;
    } else if (expecting_source_name) {
      strcpy(source_name,string);
      expecting_source_name=0;
    } else {
      sprintf(message,"read_header - unknown parameter: %s\n",string);
      fprintf(stderr,"ERROR: %s\n",message);
      exit(1);
    } 
  } 

  if (isign < 0 && OSIGN > 0){
	  fprintf(stderr,"WARNING! You are reading unsigned numbers with a signed version of dspsr/sigproc\n");
  }
  if (isign > 0 && OSIGN < 0){
	  fprintf(stderr,"WARNING! You are reading signed numbers with a unsigned version of dspsr/sigproc\n");
  }



  /* return total number of bytes read */
  return ftell(inputfile);
}

