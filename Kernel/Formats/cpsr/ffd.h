/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
/*
 * Default board ID's and serial port interfaces.
 */

#include <unistd.h>

#if cbr
#define	FFD_BOARD_ID	'a'
#define	FFD_PORT	"/dev/ttya"
#elif cbrl
#define	FFD_BOARD_ID	'a'
#define	FFD_PORT	"/dev/ttya"
#elif cpsr
#define	FFD_BOARD_ID	'a'
#define	FFD_PORT	"/dev/ttya"
#elif tmfe
#define	FFD_BOARD_ID	'a'
#define	FFD_PORT	"/dev/ttya"
#else
#endif

/*
 * Physical board map from top to bottom:
 *
 * VME	Poln	Quad	Conn	Schematic	User(SSB)	User(DSB)
 * ---	----	----	----	---------	---------	---------
 *		I	J12	ch D		ch 0		ch 0
 *	LCP
 *		Q	J11	ch C		ch 1
 * P1
 *		I	J10	ch B		ch 2		ch 2
 *	RCP
 *		Q	J9	ch A		ch 3
 *
 *
 *		I	J8	ch H		ch 4		ch 4
 *	LCP
 *		Q	J7	ch G		ch 5
 * P2 
 *		I	J6	ch F		ch 6		ch 6
 *	RCP 
 *		Q	J5	ch E		ch 7
 *
 */


/*
 * Maximum number of channels that can be in the data stream, i.e.,
 * 4 channel 2 bit mode.
 */
#define	MAXCHAN		4	/* Maximum number of channels per	*/
				/* bitpacker (i.e. host connection)	*/
#define	BOARDCHAN	8	/* Number of channels on a full board	*/
#define	DATA_MIN	2	/* minimum sample size (bits)		*/
#define	DATA_MAX	8	/* maximum sample size (bits)		*/
#define MAXWORD		16	/* maximum data word size (bits),	*/
				/* word size = nchan x nbits, e.g.,	*/
				/* 4chan x 4bits or 2chan x 8bits	*/



/*
 * Valid board address range.
 */
#define	FFD_BOARD_MIN	'a'
#define	FFD_BOARD_MAX	'd'



#ifndef SET_NONBLOCK
#define   SET_NONBLOCK(f)  fcntl(f, F_SETFL, fcntl(f, F_GETFL, 0)|O_NONBLOCK)
#define   CLR_NONBLOCK(f)  fcntl(f, F_SETFL, fcntl(f, F_GETFL, 0)&~O_NONBLOCK)
#endif



#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef STRLEN
#define STRLEN 80
#endif



typedef struct {
    float       re;
    float       im;
} COMPLEX;


#ifdef __cplusplus
extern "C" {
#endif

int str2byte( char *arg );
long str2long( char *arg );
void voltage_unpack( u_char *raw, void *data, int channel, int nchan, int bits,
		     int npts, int SSB, int VOLTS, int FLOAT );

int serial_read( u_char *value );
int serial_write( u_char *data, int npts );
int ffd_dac_update( u_char dac_chn, u_char nwr );
int ffd_header( int channel, char *header, int DEBUG );
int ffd_gain( int channel, double *db, int DEBUG );
int ffd_gain_dac_volts( int channel, double vneg, double vpos, int DEBUG );
int ffd_level( int chn, double *level_neg, double *level_pos, int DEBUG );
int ffd_mode( int bits, char *list, int DEBUG );
int ffd_register_read( char reg, u_char *value );
int ffd_register_write( char reg, u_char value );
void ffd_reset( char *device );
void ffd_serial_setup( char *device );
void ffd_serial_close();
void ffd_board( char board_id );
int ffd_start( int DEBUG );
int ffd_stop( int DEBUG );
int ffd_state();
int ffd_test();
void ffd_verbose( int VERBOSE );
void ffd_missing( int DEBUG );
void ffd_prompt();
int ffd_tmfe_multiplex();
int ffd_tmfe_level_set( int channel_lower, int channel_upper );

#ifdef __cplusplus
}
#endif
