/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef _GMRT_PARAMS_H
#define _GMRT_PARAMS_H

struct gmrt_params {
    int n_ds;
    int n_chan;
    double ch_bw;
    double rf;
    int band_dir;
    char psr_name[12];
    double dm;
    int fft_len;
    int overlap;
    int n_bins;
    float t_dump;
    int n_dump;
    long long n_samp_dump;
    long imjd;   /* Note:  These are NOT precise obs start times, */
    double fmjd; /*        just a rough estimate to check that    */
                 /*        the polycos are valid                  */
    int cal_scan;

    char scan[256];
    char observer[256];
    char proj_id[256];
    char comment[1024];
    char telescope[2];
    char front_end[256];
    char pol_mode[12];
    double ra;
    double dec;
    float epoch;
};

struct gmrt_ds_params {
    char ds_name[64];  /* This DS's hostname */
    int n_slaves;      /* Number of slaves to send to */
    int chan_offset;   /* Offset relative to full set of channels */
};

int get_params(struct gmrt_params *obs_params);
int get_ds_params(struct gmrt_ds_params *ds_params, char *nodes[], int *zap);
int get_ds_list(char *ds[]);

#endif

