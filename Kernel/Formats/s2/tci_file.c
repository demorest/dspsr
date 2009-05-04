/***************************************************************************
 *
 *   Copyright (C) 2002 by Craig West
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*
$Id: tci_file.c,v 1.4 2009/05/04 23:17:13 straten Exp $
$Log: tci_file.c,v $
Revision 1.4  2009/05/04 23:17:13  straten
verbosity mods

Revision 1.3  2007/01/24 21:43:41  straten
return error code when hdr_size is incorrect

Revision 1.2  2006/07/09 13:31:15  wvanstra
copyright notice added: Academic Free License

Revision 1.1  2002/05/01 01:53:10  cwest
s2 specific code separated from s2/s2tci

Revision 1.2  1998/08/06 02:06:54  wvanstra
cleaned up lots of -Wall warnings

 * Revision 1.1.1.1  1998/08/05  08:36:35  wvanstra
 * start
 *
Revision 1.2  1997/04/10 22:23:52  sasha
*** empty log message ***

Revision 1.1  1996/11/18 19:53:34  sasha
Initial revision

Revision 1.1  1996/09/01 17:59:21  willem
Initial revision

*/


#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <ctype.h>

#include "tci_file.h"    /* includes - tci_def.h which includes 'rcl.h' */

int tci_file_verbose = 0;

/* The following functions deal with writing and reading TCI information to
   and from a disk */

/*****************************************************************************/
int tci_file_open (const char* filename, tci_fd* tci_file,
		   tci_hdr* header, char rdwt)
/* This function opens a file of the name 'filename' for reading or writing.  
   Reading or writing is determined by the character 'rdwt' parameter. Allowed
   values of this parameter are:
                  - 'r' to open a TCI file for reading
                  - 'w' to open a NEW TCI file for writing
                  - 'a' to add to an old TCI file (not necessarily at the end)
                  - 'o' to overwrite a TCI file.  Not to be used unless 'OK'd 
                        by the user.

   When reading:  The function strips the TCI formatted header off of the file 
   and returns it in the parameter 'header'.  As well, the file size, in the 
   number of CYCLES, is returned if there is no error.  This file size is the
   actual size of the file less the size of the header (so in fact is the 
   amount of actual data in  the file).

   When writing:  The function simply creates a file with the name: 'filename'
   If this file already exists, the function returns the appropriate UNIX
   error code.  If you wish to overwrite the file, then recall this
   function with the -o option.

   The header information is written to the file and the function returns.  If
   the information in the header is invalid at the time of opening the file, it
   can be overwritten at a later time using `tci_file_hdr_write'
   HOWEVER, THE HEADER MUST CONTAIN, IN THE 'hdr_size' FIELD, THE SIZE OF THE 
   HEADER IN BYTES.
  
   The space in the file will be reserved, and cannot be changed afterward!!
   Data can be written to the file using calls to 'tci_file_buf_write'. 
   Later, when the header information is made available, it can be written
   to file using 'tci_file_header_write' with the same sized header as that
   with which the file was initially created.
   If no error occurs, the function will return 0.  (When opening for reading,
   it returns the data size, as mentioned above).

   In both cases: (both reading and writing) The unix file descriptor is
   returned via
   'tci_file'.  Upon encountering an error, an error code is
   returned by the function.
 */
{
  tci_hdr      local_header;
  struct stat  file_info;
  int          wr_mode = O_WRONLY | O_CREAT | O_TRUNC;
  int          rd_mode = O_RDWR;
  int          tci_err;

  if (filename == NULL)
    return -1;
  if (tci_file == NULL)
    return -1;

  tci_file_hdr_init (&local_header);
  if (header == NULL)
    header = &local_header;

  switch (rdwt)  {

   case 'r':
     rd_mode = O_RDONLY;
   case 'a':
     if (stat (filename, &file_info) < 0)
     {
       if (tci_file_verbose)
       {
  	 fprintf (stderr, "TCI_FILE_OPEN: (r) file: %s\n",filename);
	 perror ("TCI_FILE_OPEN:  Error");
       }
       return (-1);
     }
     
     tci_file->fsz = (u_long)file_info.st_size;
     if (tci_file->fsz  < TCI_HEADER_BASE_SIZE)
     {
       if (tci_file_verbose)
         fprintf (stderr, "TCI_FILE_OPEN: (r) file smaller than header.\n");
       return (-1);
     }

     if ((tci_file->fd = open (filename, rd_mode)) < 0)
     {
       if (tci_file_verbose)
       {
         fprintf (stderr, "TCI_FILE_OPEN: (r) can't open file: %s\n",filename);
	 perror ("TCI_FILE_OPEN:  Error");
       }
       return (-1);
     }

     /* Check that the header seems like a TCI header */
     if ( read(tci_file->fd, (char*)header, TCI_HEADER_BASE_SIZE) < 0 )
     {
       if (tci_file_verbose)
         perror ("TCI_FILE_OPEN: (r) File header could not be read ");
       close (tci_file->fd);
       return (-1);
     }
     fromBigE(&(header->hdr_size), sizeof(int));

     if ( header->hdr_size < TCI_HEADER_BASE_SIZE )
     {
       if (tci_file_verbose)
       {
         fprintf (stderr, "TCI_FILE_OPEN: (r) file header too small.\n");
	 fprintf (stderr, "TCI_FILE_OPEN: hdr sz = %i.\n",header->hdr_size);
       }
       return -1;
     }
     
     if ( header->hdr_size > tci_file->fsz )
     {
       if (tci_file_verbose)
       {
         fprintf (stderr, "TCI_FILE_OPEN: (r) Hdr_Sz larger than file.\n");
	 fprintf (stderr, "TCI_FILE_OPEN: hdr sz = %i.\n",header->hdr_size);
       }
       return -1;
     }
     
     /* set the values in tci_file... these help simplify seeking thru the
	file and ensuring that the data in the file isn't comprimised by
	data of a different rate */
     tci_file->base = header->hdr_size;
     tci_file->data_rate = (u_int)(header->hdr_drate) * 1000000;
     tci_file->fsz -= header->hdr_size;
     tci_file->fsz /= 2;
     
#ifdef TCI_FILE_DEBUG
     fprintf (stderr,"TCI_FILE_OPEN: (a) File opened with stats:\n");
     fprintf (stderr,"   header size: %i Bytes\n",tci_file->base);
     fprintf (stderr,"   data rate:   %i Wps\n",tci_file->data_rate);
     fprintf (stderr,"   data size:   %lu Words\n",tci_file->fsz);
#endif
     
     /* in the case that the header size is greater than TCI_HEADER_BASE_SIZE,
	we must ensure that the file pointer is positioned at the beginning
	of the data */
     tci_err = tci_file_sec_set (*tci_file, 0);
     if (tci_err != 0)
     {
       if (tci_file_verbose)
         perror ("TCI_FILE_OPEN: (r) Could not set file pointer.\n");
       return(tci_err);
     }
     return (0);
      
   case 'w':
     wr_mode |= O_EXCL; /* TRY */
   case 'o':
     if (!(header->hdr_drate))
     {
       if (tci_file_verbose)
         fprintf (stderr, "TCI_FILE_OPEN: (w) Data rate not defined in hdr\n");
       return (-1);
     }
     tci_file->fd = open(filename, wr_mode, TCI_FILE_PERMISSION);
     if ((tci_file->fd < 0) && (errno == EEXIST) && (rdwt == 'o'))
     {
       if (remove (filename) < 0)
       {
	 if (tci_file_verbose)
	 {
	   fprintf (stderr, "TCI_FILE_OPEN: (%c) Could not remove file: %s\n",
		    rdwt, filename);
	   perror("TCI_FILE_OPEN:  Error");
	 }
	 return (-1);
       }
       /* wr_mode &= ~O_EXCL; */
       tci_file->fd = open(filename, wr_mode, TCI_FILE_PERMISSION);
     }
     if (tci_file->fd < 0)
     {
       if (tci_file_verbose)
       {
         fprintf (stderr, "TCI_FILE_OPEN: (%c) Could not open file: %s\n",
		rdwt, filename);
	 perror("TCI_FILE_OPEN:  Error");
       }
       return(-1);
     }
     
     if (!(header->hdr_size))
       header->hdr_size = TCI_HEADER_BASE_SIZE;
     tci_file->base = header->hdr_size;
     tci_file->data_rate = (u_int)(header->hdr_drate) * 1000000;
     
     toBigE(&(header->hdr_size), sizeof(int));
     if (write(tci_file->fd,(char*)header,tci_file->base) < tci_file->base)
     {
       if (tci_file_verbose)
         perror ("TCI_FILE_OPEN:  (w) Could not write header to file ");
       return (-1);
     }
     fromBigE(&(header->hdr_size), sizeof(int));
     
#ifdef TCI_FILE_DEBUG
     printf ("TCI_FILE_OPEN: (w) File stats written to disk:\n");
     printf ("   header size: %i Bytes\n",header->hdr_size);
     printf ("   data rate:   %i Mwps\n",header->hdr_drate);
     printf ("TCI_FILE_OPEN:  File successfully opened for writing.\n");
#endif
     return (0);


   default:
     {
       if (tci_file_verbose)
	 fprintf (stderr, "TCI_FILE_OPEN:  Unknown open code.\n");
       return (-1);
     }
   }
}

int tci_file_close (tci_fd tci_file)
{
   if (close (tci_file.fd) < 0)
      return (-1);
   return (0);
}


int tci_file_buf_write ( tci_fd tci_file, u_short* dat_buf, u_int buf_sz )
/* This function writes 'buf_sz' cycles to disk.
   A TCI cycle is two bytes corresponding with the tvg word 
   of 16-bits as well as the 16-bit transfer width of the DMA card.
   The tci_fd descriptor MUST have been opened with 'tci_file_open'
   with the 'w' mode passed to 'rdwt'.  On success, this function 
   returns the number of cycles read.  On error, it returns an error code.
 */
{
   int bytes_written;

#ifdef TCI_FILE_DEBUG
      printf ("TCI_FILE_BUF_WRITE:  Writing %i bytes to file.\n", buf_sz*2 );
#endif

   if (buf_sz <= 0)  {
#ifdef TCI_FILE_DEBUG
         printf ("TCI_FILE_BUF_WRITE:  Invalid buf_sz: %i.\n", buf_sz );
#endif
      return (-1);
   }

   bytes_written = write (tci_file.fd, (char*)dat_buf, buf_sz * 2);
   if (bytes_written < (buf_sz * 2))  {
      perror ("TCI_FILE_BUF_WRITE:  Could not write buffer to file ");
      return (-1);
   }

#ifdef TCI_FILE_DEBUG
      printf ("TCI_FILE_BUF_WRITE:  Successfully wrote %i bytes to file.\n",
	      bytes_written );
#endif
   return (bytes_written/2);
}

/*****************************************************************************/
int tci_file_buf_read ( tci_fd tci_file, u_short* dat_buf, u_int buf_sz )
/* This function reads the next 'buf_sz' cycles from disk.  A TCI cycle is two 
   bytes corresponding with the tvg word of 16-bits as well as the 16-bit 
   transfer width of the DMA card.  The tci_fd descriptor MUST have been 
   opened with 'tci_file_open' with the 'w' mode passed to 'rdwt'.  On success,
   this function returns the number of cycles read.  On error, it returns an 
   error code.
 */
{
   int bytes_read;

   if (buf_sz <= 0)  {
#ifdef TCI_FILE_DEBUG
     printf ("TCI_FILE_BUF_READ:  Invalid buf_sz: %i.\n", buf_sz );
#endif
     return (-1);
   }
   
#ifdef TCI_FILE_DEBUG
   printf ("TCI_FILE_BUF_READ:  Reading %i bytes from file.\n", buf_sz*2 );
#endif
   
   if ((bytes_read=read (tci_file.fd, (char*)dat_buf, buf_sz*2)) < buf_sz*2)  {
     perror ("TCI_FILE_BUF_READ:  Could not read buffer from file ");
     return (-1);
   }
   
#ifdef TCI_FILE_DEBUG
   printf ("TCI_FILE_BUF_READ:  Successfully read %i bytes from file.\n",
	   bytes_read );
#endif
   return (bytes_read/2);
}

/* *************************************************************************
   tci_util_flush_header - clears a header with a tame sequence of spaces.
   ************************************************************************* */
int tci_file_hdr_init  (tci_hdr *header)
{
  int i;
  
  header->hdr_size = 0;
  header->hdr_drate = 0;
  sprintf (header->hdr_time, "%-*.*s", (TCI_HEADER_BASE_SIZE-6),
	   (TCI_HEADER_BASE_SIZE-6), " ");
  header->hdr_time[0] = '\0';
  header->hdr_s2mode[0] = '\0';
  header->hdr_tapeid[0] = '\0';
  for (i=0; i<4; i++)
    header->hdr_labels[i][0] = '\0';
  header->hdr_usr_field1[0] = '\0';  
  header->hdr_usr_field2[0] = '\0'; 
  header->hdr_usr_field3[0] = '\0';
  header->hdr_usr_field4[0] = '\0';
  
  return (0);
}


/***************************************************************************/
int tci_file_hdr_write ( tci_fd tci_file, tci_hdr header )
{
   off_t   fp_store;

   fp_store = lseek (tci_file.fd, 0L, SEEK_CUR);
   if (fp_store < 0)  {
      perror ("TCI_FILE_HDR_WRITE:  Could not store current file pointer ");
      return (-1);
   }
   if (lseek (tci_file.fd, 0L, SEEK_SET) < 0)  {
      perror ("TCI_FILE_HDR_WRITE:  Could not set file pointer to beginning "); 
      return (-1);
   }
   toBigE(&(header.hdr_size), sizeof(int));

   if (write (tci_file.fd, (char*)(&header), TCI_HEADER_BASE_SIZE) < 
                                                     TCI_HEADER_BASE_SIZE)  {
      perror ("TCI_FILE_HDR_WRITE:  Could not write header to file ");
      return (-1);
   }
   if (lseek (tci_file.fd, fp_store, SEEK_SET) < 0)  {
      perror ("TCI_FILE_HDR_WRITE:  Could not reset file pointer ");
      return (-1);
   }
   return(0);
}

/***************************************************************************/
int tci_file_hdr_read ( tci_fd tci_file, tci_hdr* header )
{
   off_t   fp_store;

   fp_store = lseek (tci_file.fd, 0L, SEEK_CUR);
   if (fp_store < 0)  {
      perror ("TCI_FILE_HDR_READ:  Could not store current file pointer ");
      return (-1);
   }
   if (lseek (tci_file.fd, 0L, SEEK_SET) < 0)  {
      perror ("TCI_FILE_HDR_READ:  Could not set file pointer to beginning "); 
      return (-1);
   }

   if (read (tci_file.fd, (char*)(header), TCI_HEADER_BASE_SIZE)
                                                   < TCI_HEADER_BASE_SIZE)  {
      perror ("TCI_FILE_HDR_READ:  Could not read header from file ");
      return (-1);
   }
   fromBigE(&(header->hdr_size), sizeof(int));

   if (lseek (tci_file.fd, fp_store, SEEK_SET) < 0)  {
      perror ("TCI_FILE_HDR_READ:  Could not reset file pointer ");
      return (-1);
   }
   return(0);
}


/***************************************************************************/
int tci_file_sec_set (tci_fd tci_file, u_int second)
/* move file pointer to position = second */
{
/*
tci_file.fd 		- the file descriptor
tci_file.base		- file pointer at the beginner ?
tci_file.data_rate	- 16 bit words in 1 sec interval 
tci_file.fsz		- file size in 16 bit words
*/
   off_t fp;

#ifdef TCI_FILE_DEBUG
   fprintf (stderr, "TCI_FILE_SET_SECOND: set to sec #%u\n",second);
   fprintf (stderr, "TCI_FILE_SET_SECOND: data rate %u\n",tci_file.data_rate);
   fprintf (stderr, "TCI_FILE_SET_SECOND: base addr %u\n",tci_file.base);
#endif
   
   fp= lseek(tci_file.fd,(second*tci_file.data_rate*2)+tci_file.base,SEEK_SET);
   if (fp < 0)  {
     perror ("TCI_FILE_SET_SECOND:  Could not set seek pointer");
      return (-1);
   }

#ifdef TCI_FILE_DEBUG
   fprintf (stderr, "TCI_FILE_SET_SECOND: set to byte #%li\n",fp);
#endif
   return (0);
}


/**************************************************************************/
int tci_file_sec_advance (tci_fd tci_file, u_int num_secs)
{
   int fp;

   fp = lseek (tci_file.fd, (num_secs * tci_file.data_rate * 2), SEEK_CUR);
   if (fp < 0)  {
      perror ("TCI_FILE_SORT_ADVANCE:  Could not set seek pointer");
      return (-1);
   }
   return (0);
}


/* *************************************************************************
   numeric - tests if every character in a string is numeric
   ************************************************************************* */
int numeric (char* test_string)
{
  int str_length;
  int i;

  if (!test_string)
    return (0);
  if (!(str_length = strlen(test_string)))
    return (0);
  for (i=0; i< str_length; i++)
    if ( !isdigit((int)test_string[i]) )
      return (0);
  return (1);
}

int tci_file_header_display (tci_hdr header, char* format)
{
   char*  disp_time = NULL;

   printf ("File Data Rate     %u MW/s\n", (u_int)header.hdr_drate);
   printf ("Header Size        %i B\n\n",    header.hdr_size);

   if (!numeric(header.hdr_time))
      disp_time = strdup ("no time in header\n");
   else  {
      disp_time = strdup (header.hdr_time);
   }
   printf ("File Start Time    %s\n", disp_time);
   free (disp_time);

   printf ("Recorder Mode      %s\n", header.hdr_s2mode);
   printf ("Tape ID            %s\n\n", header.hdr_tapeid);

   printf ("______________User_info______________________\n");
   printf ("#    LABEL                    INFO\n\n");

   printf ("1 %-*s: %s\n", 23,
                        header.hdr_labels[0], header.hdr_usr_field1);
   printf ("2 %-*s: %s\n", 23,
                        header.hdr_labels[1], header.hdr_usr_field2);
   printf ("3 %-*s: %s\n", 23,
                        header.hdr_labels[2], header.hdr_usr_field3);
   printf ("4 %-*s: %s\n\n",23,
                        header.hdr_labels[3], header.hdr_usr_field4);
   return (0);
}

void endian_convert (void *num, int nbytes)
{
  unsigned char tmp[8];
  unsigned char *numPtr = (unsigned char *)num;
  int i;
  
  for (i=0; i < nbytes; i++)
    tmp[i] = numPtr[nbytes-i-1];
  
  memcpy((void *)num, (void *)tmp, nbytes);
}
