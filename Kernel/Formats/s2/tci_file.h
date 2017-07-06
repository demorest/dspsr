/***************************************************************************
 *
 *   Copyright (C) 2002 by Craig West
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*
$Id: tci_file.h,v 1.3 2009/05/04 23:17:13 straten Exp $
$Log: tci_file.h
Revision 1.3  2009/05/04 23:17:13  straten
verbosity mods

Revision 1.2  2006/07/09 13:27:09  wvanstra
copyright notice added: Academic Free License

Revision 1.1  2002/05/01 01:53:10  cwest
s2 specific code separated from s2/s2tci

Revision 1.1.1.1  1998/08/05 08:36:36  wvanstra
start

Revision 1.1  1996/11/18 19:53:35  sasha
Initial revision

Revision 1.1  1996/09/01 17:59:23  willem
Initial revision

*/


#ifndef TCI_FILE_DEFD
#define TCI_FILE_DEFD

/* These following lines define the composition of the S2-TCI data file
   header. */

#define TCI_TIME_FORMAT     "yyyydddhhmmss"  /* this is the form of the 
                                                string saved in the header
                                                "hdr_time" member */
#define TCI_TIME_STRLEN     14   /* length of the above string (inc. NULL)  */
#define TCI_MODE_STRLEN     21
#define TCI_TAPEID_STRLEN   21
#define TCI_LABEL_STRLEN    17   /* user defined field labels from RCL link */
#define TCI_FIELD1_2_STRLEN 17   /* user fields 1 and 2 from RCL            */
#define TCI_FIELD3_STRLEN   33   /* 3rd user field from the RCL             */
#define TCI_FIELD4_STRLEN   49   /* 4th user field from the RCL             */
#define TCI_STATNUM_STRLEN  4    /* the station number from the RCL         */
#define TCI_SERLNUM_STRLEN  6    /* the serial number from the RCL          */
#define TCI_NICKNAME_STRLEN 9    
/* The basic length of a header is the sum of all its parts... the first four
   bytes are a big endian integer number, representing the size of the file 
   file header in bytes.
 */
#define TCI_HEADER_BASE_SIZE (  4                       \
                              + 1                       \
                              + TCI_TIME_STRLEN         \
                              + TCI_MODE_STRLEN         \
                              + 1                       \
                              + TCI_TAPEID_STRLEN    \
                              + 4 * TCI_LABEL_STRLEN    \
                              + 2 * TCI_FIELD1_2_STRLEN \
                              + TCI_FIELD3_STRLEN       \
                              + TCI_FIELD4_STRLEN       )

/* Un-comment these lines if you decide to include these fields in the header
                        + TCI_STATNUM_STRLEN            \
                        + TCI_SERLNUM_STRLEN            \
                        + TCI_NICKNAME_STRLEN )
*/

/* *************************************************************************
   The s2tci file header structure
   ************************************************************************* */
typedef struct {
   int  hdr_size;     /* the size of the header in bytes */  
   char hdr_drate;    /* a code for the data rate of transmission...
                      It is the number of Mega bits per second (Mbps)
                      transmitted by the S2, divided by 16.  (In other words,
                      the number of Mega words per second [Mwps]).  In essence,
                      it can also be thought of as the number of transports
                      used during the transfer.  */

   char hdr_time          [TCI_TIME_STRLEN];
   char hdr_s2mode        [TCI_MODE_STRLEN];
   char reserved; 
   char hdr_tapeid        [TCI_TAPEID_STRLEN];
   char hdr_labels[4]     [TCI_LABEL_STRLEN];
   char hdr_usr_field1    [TCI_FIELD1_2_STRLEN];
   char hdr_usr_field2    [TCI_FIELD1_2_STRLEN];
   char hdr_usr_field3    [TCI_FIELD3_STRLEN];
   char hdr_usr_field4    [TCI_FIELD4_STRLEN];

/* Un-comment these lines if you decide to include these fields in the header
   char hdr_station_num   [TCI_STATNUM_STRLEN];
   char hdr_serial_num    [TCI_SERLNUM_STRLEN];
   char hdr_nickname      [TCI_NICKNAME_STRLEN];  */

} tci_hdr;

/* *************************************************************************
   The s2tci data file structure
   ************************************************************************* */
typedef struct {
   int    fd;          /* the file descriptor */
   int    base;  /* the base address of the file pointer after which
                          data starts */
   unsigned int  data_rate;  /* the number of 2-byte words in each second */
   unsigned long fsz;        /* the size of this file in the number of 2-byte
		          words (usually only returned by tci_file_open) */
} tci_fd;

#define TCI_FILE_PERMISSION S_IWUSR | S_IRUSR | S_IROTH | S_IRGRP | S_IWGRP

#ifdef __cplusplus
extern "C" {
#endif

int tci_file_open      (const char* filename, tci_fd* tci_file, 
                        tci_hdr* header, char rdwt);

int tci_file_close     (tci_fd tci_file);
int tci_file_buf_read  (tci_fd tci_file, unsigned short* dat_buf, unsigned sz);
int tci_file_buf_write (tci_fd tci_file, unsigned short* dat_buf, unsigned sz);

int tci_file_hdr_init  (tci_hdr *header);
int tci_file_hdr_read  (tci_fd tci_file, tci_hdr* header);
int tci_file_hdr_write (tci_fd tci_file, tci_hdr header);

int tci_file_sec_set     (tci_fd tci_file, unsigned second);
int tci_file_sec_advance (tci_fd tci_file, unsigned num_secs);

int tci_file_header_display (tci_hdr header, char* format);
int tci_file_header_flush (tci_hdr *header);

void endian_convert(void *, int);

extern int tci_file_verbose;

#ifdef __cplusplus
	   }
#endif

#ifdef __alpha
#ifndef LITTLE_ENDIAN
#define LITTLE_ENDIAN
#endif
#endif

#ifdef LITTLE_ENDIAN
#define toBigE(p,s) 	endian_convert(p,s)
#define fromBigE(p,s)	endian_convert(p,s)
#else
#define toBigE(p,s)
#define fromBigE(p,s)		
#endif

#endif /* not TCI_DEF_DEFD */
