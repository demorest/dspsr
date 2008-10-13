/***************************************************************************
 *
 *   Copyright (C) 2006 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef _ASP_PARAMS_H
#define _ASP_PARAMS_H

struct asp_params {
    int32_t n_ds;
    int32_t n_chan;
    double ch_bw;
    double rf;
    int32_t band_dir;
    char psr_name[12];
    double dm;
    int32_t fft_len;
    int32_t overlap;
    int32_t n_bins;
    float t_dump;
    int32_t n_dump;
    int64_t n_samp_dump;
    int32_t imjd;   /* Note:  These are NOT precise obs start times, */
    double fmjd; /*        just a rough estimate to check that    */
                 /*        the polycos are valid                  */
    int32_t cal_scan;

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
} __attribute__ ((aligned (4), packed)); // Align as in 32-bit systems
                                         // Could use "#pragma pack" instead...

struct asp_ds_params {
    char ds_name[64];  /* This DS's hostname */
    int32_t n_slaves;      /* Number of slaves to send to */
    int32_t chan_offset;   /* Offset relative to full set of channels */
};

int get_params(struct asp_params *obs_params);
int get_ds_params(struct asp_ds_params *ds_params, char *nodes[], int *zap);
int get_ds_list(char *ds[]);

#endif

