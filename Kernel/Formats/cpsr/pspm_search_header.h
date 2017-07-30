/***************************************************************************
 *
 *   Copyright (C) 1999 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
/*
 * PSPM search header definition.
 *
 * Be VERY careful about making changes to this file as it may
 * compromise your ability to read existing data headers from
 * old data.
 *
 * To maintain portability it is explicitly assumed that:
 * double = 8Bytes,
 * long = 4Bytes,
 * and that doubles start on an even word boundary.
 *
 *
 * For the paranoid, add the following code fragment to the beginning
 * of your programs:
 *
 * if ( sizeof(PSPM_SEARCH_HEADER) != PSPM_HEADER_SIZE )
 *      some error message;
 *
 *
 * $Log: pspm_search_header.h
 * Revision 1.5  2009/06/17 10:16:54  straten
 * use ISO C99 integer types directly
 *
 * Revision 1.4  2006/11/11 20:56:38  straten
 * compile against psrchive 6.0
 *
 * Revision 1.3  2006/07/09 13:27:06  wvanstra
 * copyright notice added: Academic Free License
 *
 * Revision 1.2  2000/01/24 02:32:59  wvanstra
 * Updated for August 1999 changes to pspm_search_header
 *
 * Revision 1.1  1999/10/28  17:59:09  wvanstra
 * from baseband/timeseries
 *
 * Revision 2.2  1999/10/26  16:36:43  wvanstra
 * moved from include/ repository
 *
 * Revision 1.2  1999/01/27 07:03:27  wvanstra
 * added declaration for PSPMdisplay function
 *
 * Revision 1.1  1998/12/22  12:37:57  wvanstra
 * CPSR header files from Stuart, modified by Willem
 *
 * Revision 2.0  1998/09/08  06:35:27  mbritton
 * version 2.0
 *
 * Revision 1.2  1998/08/26  08:04:57  wvanstra
 * lots of getting CPSR to work with psrdisp
 *
 * Revision 1.1  1998/08/19 15:41:39  pulsar
 * ffd* class.h pspm* are Caltech code...  the rest are integration changes
 *
 * Revision 1.3  1998/08/03 20:31:16  paulr
 * Added MAX_ to NUM_LO_BOARDS and NUM_MF_BOARDS
 *
 * Revision 1.2  1998/07/17 00:04:12  sba
 * Switched to using rcsid.h macros to put RCS Id strings in executables
 *
 * Revision 1.1  1998/02/25 01:09:39  sba
 * Initial revision
 *
 *
 */

#include <stdio.h>

#ifndef _PSPM_SEARCH_HEADER_H
#define	_PSPM_SEARCH_HEADER_H

#include "class.h"


#ifndef PSPM_HEADER_SIZE
#define PSPM_HEADER_SIZE	(1<<15)
#endif


#define	DRIFT_TYPE	0
#define	POINT_TYPE	2

#include "environ.h"
#include "quadruple.h"

typedef struct {

#if PSPM

    double samp_rate;
    double pasmon_az;
    double pasmon_za;
    double user_az;
    double user_za;
    double pasmon_lmst;		/* local mean siderial time in seconds */
    double rf_freq;		/* Sky Frequency which is down converted to  */
				/* IF frequency PSPM_IF (MHz)		     */
    double tick_offset;
    double bw;
    double length_of_integration; /* some fixed number */
    long header_version;
    long scan_file_number;
    long bit_mode;
    long scan_num;
    long tc;
    long num_chans;
    long pasmon_wrap;
    long pasmon_feed;
    long pasmon_daynumber;
    long pasmon_ast;		/* hhmmss */
    char psr_name[12];
    char date[12];
    char start_time[12];
    long file_size;
    long tape_num;
    long tape_file_number;
    char obs_group[12];
    char even_word_boundary_filler[4];
    double user_ra;		/* J2000 (10000.*hr+100.*min+sec)	    */
    double user_dec;		/* J2000 (10000.*deg+100.*min+sec)	    */
    double chan_first_freq;	/* IF center frequency of first channel (MHz)*/
    double chan_spacing;	/* Spaceing between adjacent channel center */
				/* frequencies (MHz)			    */
    int	SIDEBAND;		/* sideband				    */
    char filler[32536];
    long BACKEND_TYPE;
    long UPDATE_DONE;
    long HEADER_TYPE;

#elif CBR

    double samp_rate;
    double pasmon_az;
    double pasmon_za;
    double user_az;
    double user_za;
    double pasmon_lmst;		/* local mean siderial time in seconds */
    double rf_freq;		/* Sky Frequency which is down converted to  */
				/* IF frequency PSPM_IF (MHz)		     */
    double tick_offset;
    double bw;
    double length_of_integration; /* some fixed number */
    int32_t header_version;
    int32_t scan_file_number;
    int32_t bit_mode;
    int32_t scan_num;
    int32_t tc;
    int32_t num_chans;
    int32_t pasmon_wrap;
    int32_t pasmon_feed;
    int32_t pasmon_daynumber;
    int32_t pasmon_ast;		/* hhmmss */
    char psr_name[12];
    char date[12];
    char start_time[12];
    int32_t file_size;
    int32_t tape_num;
    int32_t tape_file_number;
    char obs_group[12];
    char even_word_boundary_filler[4];
    double user_ra;		/* J2000 (10000.*rah+100.*ram+ras)	    */
    double user_dec;		/* J2000 (10000.*rah+100.*ram+ras)	    */
    double chan_first_freq;	/* Unused				    */
    double chan_spacing;	/* Unused				    */
    int	SIDEBAND;		/* sideband				    */
    int observatory;		/* Observatory code			    */
    quadruple mjd_start;	/* Start time (MJD)			    */
    /*
     * (long long) file_size and offset where added August 1999 to the
     * source tree. Prior to this time, or when the production binaries
     * where updated for a particular backend, these values will hold 0
     * and you should use the (long) file_size above. Note, there was
     * no equivalent offset number prior to ll_file_offset.
     */
    int64_t ll_file_offset;       /* Cummulative size of all of the previous  */
                                /* files in this scan (Bytes) which can,    */
                                /* e.g. be used to calculate the start time */
                                /* of this file.                            */
    int64_t ll_file_size;         /* Size of this particular file (Bytes).    */

    char filler[32500];
    int32_t BACKEND_TYPE;
    int32_t UPDATE_DONE;
    int32_t HEADER_TYPE;

#elif BPP

    /*
     * Here are BPP header items
     */
    char head[16];		/* Holds "NBPPSEARCH\0" */
    long header_version;	/* Version number which is different for each backend */
    long scan_num;		/* Scan number e.g. 31096001 = Obs 1 on day 310 of 1996 */

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

    long bit_mode;		/* 4 = 4-bit power, -4 = 4-bit voltage in direct mode */
    long num_chans;		/* Calculated number of 4-bit channels in each sample */
    int lmst;			/* LMST time in seconds since 0h */
    char target_name[32];	/* Space for pulsar name or map name for survey */
    char date[16];		/* UT date which will match the scan number */
    char start_time[16];	/* UT time of the 1pps tick which started the obs*/
    long scan_file_number;	/* Which file number of the scan? */
    long file_size;		/* Size of this file */
    long tape_num;		/* Tape number */
    long tape_file_number;	/* File number on this tape */
    char obs_group[16];		/* Who did the observation (mainly for future) */

    int enabled_CBs;		/* Bitmap of enabled CBs */
    int mb_start_address;	/* Real base (8-bit) address of first CB reg read */
    int mb_end_address;		/* Read end (8-bit) address of last CB reg read */
    int mb_start_board;		/* First board ID read */
    int mb_end_board;		/* last board ID read (MB can only read seq. boards) */
    int mb_vme_mid_address;	/* Value stored in VME_MID register (usu. 00) */
    int mb_ack_enabled;		/* Boolean, did we use ACK protocol? */
    int start_from_ste;		/* Boolean, am I starting with the STE counter? */

    /*
     * CB Registers
     */
    int cb_sum_polarizations;	/* Boolean, did the CBs sum pols on-board? */
    int cb_direct_mode;		/* Boolean, did we read the CBs in direct-mode? */
    int cb_eprom_mode[MAXNUMCB]; /* Which EPROM table? (MAXNUMCB=6) */
    int cb_accum_length;	/* Contents of CB accum len regs (all CBs IDENTICAL) */
				/* cb_accum_length is TOTAL accum length, not accum_len-1 */

    /**
     * TB Registers
     */
    int tb_outs_reg;	/* OUTS_REG, turns on/off analog supply and PLLs */
    int tb_ste;		/* Value stored in STE counter */
    int tb_stc;		/* This need to be read AFTER an integration!!! */
    int  H_deci_factor;	/* Decimation factor */
    int GenStat0, GenStat1, Ack_Reg; /* HW registers, for debugging */

    /*
     * DFB Registers
     */
    /* These first three are the "logical" state of the DFBs */
    int dfb_sram_length;	/* Same for every board??? */
    int ASYMMETRIC;		/* Currently the same for all boards */
    float dfb_sram_freqs[FB_CHAN_PER_BRD]; /* Filled in by setmixer_board (8) */

    /* These three are for HW debugging, not to be used by analysis software */
    int dfb_mixer_reg[MAXNUMDFB]; /* Set by set_dfb_mixer (MAXNUMDFB=12) */
    int dfb_conf_reg[MAXNUMDFB];  /* Set by set_dfb_conf */
    int dfb_sram_addr_msb[MAXNUMDFB]; /* Set by set_dfb_conf */

    /* These are the ACTUAL Harris taps loaded into the DFBs */
    int i_hcoef[MAX_HARRIS_TAPS]; /* MAX_HARRIS_TAPS=256 */
    int q_hcoef[MAX_HARRIS_TAPS];


    /*
     * Hardware configuration
     */
    int tb_id;
    int cb_id[MAXNUMCB];
    int dfb_id[MAXNUMDFB];

    int aib_if_switch;	/* Which IF input are we using? (Same for both Pols) */

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
    int aib_serial;
    int aib_rev;
    char filler[28828];

    long BACKEND_TYPE;
    long UPDATE_DONE;
    long HEADER_TYPE;

#else

#error Unknown backend type for PSPM_SEARCH_HEADER

#endif

} PSPM_SEARCH_HEADER;


/*
 * Prototypes.
 */
int update_search_header( PSPM_SEARCH_HEADER *s_h );
void PSPMfromBigEndian  ( PSPM_SEARCH_HEADER *s_h );
void PSPMtoBigEndian    ( PSPM_SEARCH_HEADER *s_h );
int  PSPMdisplay (FILE* out, PSPM_SEARCH_HEADER* header, const char* field);

#endif /* _PSPM_SEARCH_HEADER_H */
