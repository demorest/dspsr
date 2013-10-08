/* filterbank.h - include file for filterbank and related routines */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* input and output files and logfile (filterbank.monitor) */
extern FILE *input, *output, *logfile;
extern char  inpfile[80], outfile[80];

/* global variables describing the data */
#include "header.h"
extern double time_offset;

/* global variables describing the operating mode */
extern float start_time, final_time, clip_threshold;

extern int obits, sumifs, headerless, headerfile, swapout, invert_band;
extern int compute_spectra, do_vanvleck, hanning, hamming, zerolagdump;
extern int headeronly;
extern char ifstream[8];

/* library of subroutines and functions */
#include "sigproc.h"
