/***************************************************************************
 *
 *   Copyright (C) 2004 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#ifndef __bpphdr_h_
#define __bpphdr_h_

// Pulled straight out of Duncan's sigproc-2-4 on 3 December 2004 by HSK
/*
  Modified by Willem van Straten on 20 August 2011
  int -> int32_t
  long -> int32_t
  long long -> int64_t
*/

#include <inttypes.h>

#ifndef BPP_HEADER_SIZE
#define BPP_HEADER_SIZE	(1<<15)
#endif

#define MAXNUMCB 6
#define MAXREGS 8
#define NIBPERREG 4
#define MAXNUMDFB (2*MAXNUMCB)
#define MAX_NUM_LO_BOARDS MAXNUMCB
#define MAX_NUM_MF_BOARDS (2*MAX_NUM_LO_BOARDS)
#define FB_CHAN_PER_BRD                   8
#define MAX_HARRIS_TAPS                 256
#define MAXNUMCHAN (MAXNUMDFB*FB_CHAN_PER_BRD)

#pragma pack(push,4)

typedef struct {

    /*
     * Here are BPP header items
     */
    char head[16];		/* Holds "NBPPSEARCH\0" */
    int32_t header_version;	/* Version number which is different for each backend */
    uint32_t scan_num;	/* Scan number e.g. 31096001 = Obs 1 on day 310 of 1996 */
				/* Changed to DDDYYYYSSS in Jan 2000 */

    /* These doubles are all aligned on 8-byte boundaries */
    double length_of_integration; /* if known in advance */
    double samp_rate;	/* Calculated from nbpp_sw ( in us despite name ) */
    double ra_1950;		/* Radians */
    double dec_1950;		/* Radians */
    double tele_x;		/* Carriage X position in m */
    double tele_y;		/* Carriage Y position in m */
    double tele_z;		/* Carriage Z position in m */
    double tele_inc;		/* Mirror inclination in deg */
    double Fclk;		/* Fast clock speed (Hz) */
    double Har_Clk;		/* Harris clock (H_deci_factor*bandwidth) */
    double bandwidth;		/* DSB channel bandwidth (== Sclk) */
    double dfb_gain[MAXNUMDFB];	/* Gain that was applied to generate Harris coeffs */

    /*
     * AIB Configuration
     */
    double aib_los[MAX_NUM_LO_BOARDS];
    double mf_filt_width[MAX_NUM_MF_BOARDS];
    double mf_atten[MAX_NUM_MF_BOARDS];
    double rf_lo;		/* LO frequency used in the receiver to generate the IF */

    int32_t bit_mode;		/* 4 = 4-bit power, -4 = 4-bit voltage in direct mode */
    int32_t num_chans;		/* Calculated number of 4-bit channels in each sample */
    int32_t lmst;			/* LMST time in seconds since 0h */
    char target_name[32];	/* Space for pulsar name or map name for survey */
    char date[16];		/* UT date which will match the scan number */
    char start_time[16];	/* UT time of the 1pps tick which started the obs*/
    int32_t scan_file_number;	/* Which file number of the scan? */
    int32_t file_size;		/* Size of this file */
    int32_t tape_num;		/* Tape number */
    int32_t tape_file_number;	/* File number on this tape */
    char obs_group[16];		/* Who did the observation (mainly for future) */

    int32_t enabled_CBs;		/* Bitmap of enabled CBs */
    int32_t mb_start_address;	/* Real base (8-bit) address of first CB reg read */
    int32_t mb_end_address;		/* Read end (8-bit) address of last CB reg read */
    int32_t mb_start_board;		/* First board ID read */
    int32_t mb_end_board;		/* last board ID read (MB can only read seq. boards) */
    int32_t mb_vme_mid_address;	/* Value stored in VME_MID register (usu. 00) */
    int32_t mb_ack_enabled;		/* Boolean, did we use ACK protocol? */
    int32_t start_from_ste;		/* Boolean, am I starting with the STE counter? */

    /*
     * CB Registers
     */
    int32_t cb_sum_polarizations;	/* Boolean, did the CBs sum pols on-board? */
    int32_t cb_direct_mode;		/* Boolean, did we read the CBs in direct-mode? */
    int32_t cb_eprom_mode[MAXNUMCB]; /* Which EPROM table? (MAXNUMCB=6) */
    int32_t cb_accum_length;	/* Contents of CB accum len regs (all CBs IDENTICAL) */
				/* cb_accum_length is TOTAL accum length, not accum_len-1 */

    /**
     * TB Registers
     */
    int32_t tb_outs_reg;	/* OUTS_REG, turns on/off analog supply and PLLs */
    int32_t tb_ste;		/* Value stored in STE counter */
    int32_t tb_stc;		/* This need to be read AFTER an integration!!! */
    int32_t  H_deci_factor;	/* Decimation factor */
    int32_t GenStat0, GenStat1, Ack_Reg; /* HW registers, for debugging */

    /*
     * DFB Registers
     */
    /* These first three are the "logical" state of the DFBs */
    int32_t dfb_sram_length;	/* Same for every board??? */
    int32_t ASYMMETRIC;		/* Currently the same for all boards */
    float dfb_sram_freqs[FB_CHAN_PER_BRD]; /* Filled in by setmixer_board (8) */

    /* These three are for HW debugging, not to be used by analysis software */
    int32_t dfb_mixer_reg[MAXNUMDFB]; /* Set by set_dfb_mixer (MAXNUMDFB=12) */
    int32_t dfb_conf_reg[MAXNUMDFB];  /* Set by set_dfb_conf */
    int32_t dfb_sram_addr_msb[MAXNUMDFB]; /* Set by set_dfb_conf */

    /* These are the ACTUAL Harris taps loaded into the DFBs */
    int32_t i_hcoef[MAX_HARRIS_TAPS]; /* MAX_HARRIS_TAPS=256 */
    int32_t q_hcoef[MAX_HARRIS_TAPS];


    /*
     * Hardware configuration
     */
    int32_t tb_id;
    int32_t cb_id[MAXNUMCB];
    int32_t dfb_id[MAXNUMDFB];

    int32_t aib_if_switch;	/* Which IF input are we using? (Same for both Pols) */

    /* matt add new stuff here */
    /* Additional Hardware information, 97apr25 MRD */
    int	mb_rev, mb_serial;
    int	tb_rev, tb_serial;
    int	cb_rev[MAXNUMCB], cb_serial[MAXNUMCB];
    int	dfb_rev[MAXNUMDFB], dfb_serial[MAXNUMDFB];
    int	mb_xtal_freq;
    int	mf_serial[MAX_NUM_MF_BOARDS], mf_rev[MAX_NUM_MF_BOARDS];
    int	lo_serial[MAX_NUM_LO_BOARDS], lo_rev[MAX_NUM_LO_BOARDS];

    int	mb_long_ds0;		/* lengthen DS0 on vme reads with ack enabled */
    int	dfb_sun_program[MAXNUMDFB];	/* Set by set_dfb_mode */
    int	dfb_eprom[MAXNUMDFB];		/* Set by set_dfb_mode */
    int	dfb_sram_addr[MAXNUMDFB];       /* rev 4 Set by set_dfb_conf */
    int	dfb_har_addr[MAXNUMDFB];        /* rev 4 Set by set_dfb_conf */
    int	dfb_clip_adc_neg8[MAXNUMDFB];	/* for use in DFB mixer table */
    int	dfb_shften_[MAXNUMDFB];		/* for low level Harris mode */
    int	dfb_fwd_[MAXNUMDFB];		/* for low level Harris mode */
    int	dfb_rvrs_[MAXNUMDFB];		/* for low level Harris mode */
    int	dfb_asymmetric[MAXNUMDFB];	/* what kind of taps to load ? */
    double dfb_i_dc[MAXNUMDFB];		/* set when programming the Mixer SRAM*/
    double dfb_q_dc[MAXNUMDFB];		/* set when programming the Mixer SRAM*/
    double max_dfb_freq;                /* used in picking decimations */

    /* 
     * pre 97apr25 was 29780
     * pre 97oct29 was 28844
     * pre 98jan20 was 28836
     */
    int32_t aib_serial;
    int32_t aib_rev;
    /*
     * (long long) file_size and offset where added August 1999 to the
     * source tree. Prior to this time, or when the production binaries
     * where updated for a particular backend, these values will hold 0
     * and you should use the (long) file_size above. Note, there was
     * no equivalent offset number prior to ll_file_offset.
     */
    int64_t ll_file_offset;	/* Cummulative size of all of the previous  */
				/* files in this scan (Bytes) which can,    */
				/* e.g. be used to calculate the start time */
				/* of this file.			    */
    int64_t ll_file_size;	/* Size of this particular file (Bytes).    */
    char quad_word_boundary_filler[8];

    long double mjd_start;	/* Start time (MJD)			    */

    char filler[28788];

    int32_t BACKEND_TYPE;
    int32_t UPDATE_DONE;
    int32_t HEADER_TYPE;

} BPP_SEARCH_HEADER;


#define	TIMING_TYPE		1	/* 2d array of unsigned longs	*/
#define	TIMING_LONG_TYPE	1	/* 2d array of unsigned longs	*/
#define	TIMING_FLOAT_TYPE	3	/* 2d array of floats		*/

typedef struct {
  
    /* In this part, double=8bytes, long=4bytes, if not, fix it! */

    char head[16]; /* Holds "NBPPTIME\0" */

    int32_t header_version;	/* Version number which is different for each backend */

    int32_t scan_num;		/* Scan number e.g. 31096001 = Obs 1 on day 310 of 1996 */
    

    double psr_period;
    double samp_rate;		/* Sample rate in (us). Note, in the case    */
				/* of oversampling this is the integrated    */
				/* sample rate not the hardware rate.	     */
    double psr_dm;
    double length_of_integration; /* An estimate that does NOT take into     */
				/* account the doppler corrections to	     */
				/* samp_rate.  For true integration time,    */
				/* the polyco.dat entry appended on the end  */
				/* of the scan must be consulted.	     */
    char psr_name[16];

    int32_t scan_file_number;	/* Which file number of the scan?	     */
    int32_t stc;			/* Unused, use stctime instead.		     */
    int32_t num_phase_bins;
    int32_t num_periods;		/* exact number of pulse periods integrated  */
    int32_t sizeof_polyco;
    int32_t oversample;		/* How many hardware samples are integrated  */
				/* before accumulating in the pulse profile  */
				/* (note, a 0 should be interpreted as a 1). */


    /* These doubles are all aligned on 8-byte boundaries */
    double ra_1950;		/* Radians */
    double dec_1950;		/* Radians */
    double tele_x;		/* Carriage X position in m */
    double tele_y;		/* Carriage Y position in m */
    double tele_z;		/* Carriage Z position in m */
    double tele_inc;		/* Mirror inclination in deg */
    double Fclk;		/* Fast clock speed (Hz) */
    double Har_Clk;		/* Harris clock (H_deci_factor*bandwidth) */
    double bandwidth;		/* DSB channel bandwidth (== Sclk) */
    double dfb_gain[MAXNUMDFB];	/* Gain that was applied to generate Harris coeffs */

    /*
     * AIB Configuration
     */
    double aib_los[MAX_NUM_LO_BOARDS];
    double mf_filt_width[MAX_NUM_MF_BOARDS];
    double mf_atten[MAX_NUM_MF_BOARDS];
    double rf_lo;		/* LO frequency used in the receiver to generate the IF */

    int32_t bit_mode;		/* 4 = 4-bit power, -4 = 4-bit voltage in direct mode */
    int32_t num_chans;		/* Calculated number of 4-bit channels in each sample */
    int32_t lmst;			/* LMST time in seconds since 0h */
    char target_name[32];	/* Space for pulsar name or map name for survey */
    char date[16];		/* UT date which will match the scan number */
    char start_time[16];	/* UT time of the 1pps tick which started the obs*/
    int32_t file_size;		/* Size of this file */
    int32_t tape_num;		/* Tape number */
    int32_t tape_file_number;	/* File number on this tape */
    char obs_group[16];		/* Who did the observation (mainly for future) */

    int32_t enabled_CBs;		/* Bitmap of enabled CBs */
    int32_t mb_start_address;	/* Real base (8-bit) address of first CB reg read */
    int32_t mb_end_address;		/* Read end (8-bit) address of last CB reg read */
    int32_t mb_start_board;		/* First board ID read */
    int32_t mb_end_board;		/* last board ID read (MB can only read seq. boards) */
    int32_t mb_vme_mid_address;	/* Value stored in VME_MID register (usu. 00) */
    int32_t mb_ack_enabled;		/* Boolean, did we use ACK protocol? */
    int32_t start_from_ste;		/* Boolean, am I starting with the STE counter? */

    /*
     * CB Registers
     */
    int32_t cb_sum_polarizations;	/* Boolean, did the CBs sum pols on-board? */
    int32_t cb_direct_mode;		/* Boolean, did we read the CBs in direct-mode? */
    int32_t cb_eprom_mode[MAXNUMCB]; /* Which EPROM table? (MAXNUMCB=6) */
    int32_t cb_accum_length;	/* Contents of CB accum len regs (all CBs IDENTICAL) */
				/* cb_accum_length is TOTAL accum length, not accum_len-1 */

    /**
     * TB Registers
     */
    int32_t tb_outs_reg;	/* OUTS_REG, turns on/off analog supply and PLLs */
    int32_t tb_ste;		/* Value stored in STE counter */
    int32_t tb_stc;		/* This need to be read AFTER an integration!!! */
    int32_t  H_deci_factor;	/* Decimation factor */
    int32_t GenStat0, GenStat1, Ack_Reg; /* HW registers, for debugging */

    /*
     * DFB Registers
     */
    /* These first three are the "logical" state of the DFBs */
    int32_t dfb_sram_length;	/* Same for every board??? */
    int32_t ASYMMETRIC;		/* Currently the same for all boards */
    float dfb_sram_freqs[FB_CHAN_PER_BRD]; /* Filled in by setmixer_board (8) */

    /* These three are for HW debugging, not to be used by analysis software */
    int32_t dfb_mixer_reg[MAXNUMDFB]; /* Set by set_dfb_mixer (MAXNUMDFB=12) */
    int32_t dfb_conf_reg[MAXNUMDFB];  /* Set by set_dfb_conf */
    int32_t dfb_sram_addr_msb[MAXNUMDFB]; /* Set by set_dfb_conf */

    /* These are the ACTUAL Harris taps loaded into the DFBs */
    int32_t i_hcoef[MAX_HARRIS_TAPS]; /* MAX_HARRIS_TAPS=256 */
    int32_t q_hcoef[MAX_HARRIS_TAPS];


    /*
     * Hardware configuration
     */
    int32_t tb_id;
    int32_t cb_id[MAXNUMCB];
    int32_t dfb_id[MAXNUMDFB];

    int32_t aib_if_switch;	/* Which IF input are we using? (Same for both Pols) */

    /* matt add new stuff here */
    /* Additional Hardware information, 97apr25 MRD */
    int32_t mb_rev, mb_serial;
    int32_t tb_rev, tb_serial;
    int32_t cb_rev[MAXNUMCB], cb_serial[MAXNUMCB];
    int32_t dfb_rev[MAXNUMDFB], dfb_serial[MAXNUMDFB];
    int32_t mb_xtal_freq;
    int32_t mf_serial[MAX_NUM_MF_BOARDS], mf_rev[MAX_NUM_MF_BOARDS];
    int32_t lo_serial[MAX_NUM_LO_BOARDS], lo_rev[MAX_NUM_LO_BOARDS];

    int32_t	mb_long_ds0;		/* lengthen DS0 on vme reads with ack enabled */
    int32_t	dfb_sun_program[MAXNUMDFB];	/* Set by set_dfb_mode */
    int32_t	dfb_eprom[MAXNUMDFB];		/* Set by set_dfb_mode */
    int32_t	dfb_sram_addr[MAXNUMDFB];       /* rev 4 Set by set_dfb_conf */
    int32_t	dfb_har_addr[MAXNUMDFB];        /* rev 4 Set by set_dfb_conf */
    int32_t	dfb_clip_adc_neg8[MAXNUMDFB];	/* for use in DFB mixer table */
    int32_t	dfb_shften_[MAXNUMDFB];		/* for low level Harris mode */
    int32_t	dfb_fwd_[MAXNUMDFB];		/* for low level Harris mode */
    int32_t	dfb_rvrs_[MAXNUMDFB];		/* for low level Harris mode */
    int32_t	dfb_asymmetric[MAXNUMDFB];	/* what kind of taps to load ? */
    double dfb_i_dc[MAXNUMDFB];		/* set when programming the Mixer SRAM*/
    double dfb_q_dc[MAXNUMDFB];		/* set when programming the Mixer SRAM*/
    double max_dfb_freq;                /* used in picking decimations */

    int32_t aib_serial;
    int32_t aib_rev;

    double tick_offset;		        /* delay from 1sec tick to start time (usec) */
    double startup_latency;	        /* estimated (by Matt?) time delay through */
                                        /* system (usec) */
    double initial_phase_offset;	/* Time offset to correct start time */
					/* from first sample at an arbitrary */
					/* phase to the start time of the    */
					/* first folded phase bin (sec).     */
    double stctime;			/* STC counter converted to sec.     */

    char filler[28746];
    int32_t BACKEND_TYPE;
    int32_t UPDATE_DONE;
    int32_t HEADER_TYPE;

} BPP_TIMING_HEADER;

#pragma pack(pop)

#endif // #ifndef __bpphdr_h_
