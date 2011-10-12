//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2011 by Paul Demorest
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __PdevFile_h
#define __PdevFile_h

#include "dsp/File.h"

#define PDEV_HEADER_BYTES ((size_t)1024)

namespace dsp {

  //! PdevFile reads baseband data from the Mock Spectrometers at Arecibo
  /*! Data in pdev format from the Mocks comes in a multi-file format
   *  where only the first file in the set contains a 1024-byte header.
   *  The rest of the file, and all subsequent files, contains 8-bit 
   *  complex dual pol single-channel samples.  These are in the same 
   *  format as is handled by ASPUnpacker.  
   *
   *  Since the pdev header does not contain a full description of the
   *  observation, we use an ASCIIObservation style header.  dspsr should
   *  be pointed at the ASCII header file (not the data files).  The
   *  DATAFILE keyword gives the base file name for the set of files.
   *  In a given set, file 0 must be present.  DATAFILE should 
   *  contain only the 'base' of the file names, for example:
   *
   *    p2613.20110323.b0s1g0.00000.pdev
   *
   *  The correct base name is 'p2613.20110323.b0s1g0'.
   *
   *  The first file in the set contains the obs start timestamp, 
   *  so its index must be stated via the STARTFILE keyword.
   *
   *  The ASCII file must have INSTRUMENT set to 'Mock' for this
   *  class to recognize the file.
   */
  class PdevFile : public File
  {
  public:
	  
    //! Construct and open file	  
    PdevFile (const char* filename=0, const char* headername=0);
	  
    //! Destructor
    ~PdevFile ();
	  
    //! Returns true if filename is a valid pdev file
    bool is_valid (const char* filename) const;

    //! Polns need to be swapped
    bool swap_poln () const { return rawhdr[3] & 0x02; }

    //! I/Q need to be swapped
    bool swap_dim () const { return rawhdr[3] & 0x01; }

  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Get stats about the whole set of files
    void check_file_set ();

    //! Return number of samples
    int64_t fstat_file_ndat (uint64_t tailer_bytes=0);

    //! Load bytes from the file set
    int64_t load_bytes (unsigned char *buffer, uint64_t nbytes);

    //! Seek to a certain spot in the file set
    int64_t seek_bytes (uint64_t bytes);

    //! Parse the "aoHdr" part of the binary header
    void parse_aoHdr ();

  private:

    //! Raw header
    unsigned int rawhdr[PDEV_HEADER_BYTES / 4];

    //! True if we have an ASCII header file
    bool have_ascii_header;

    //! Base file name
    char basename[256];

    //! Current file index
    int curfile;

    //! Start file index
    int startfile;

    //! Final file index
    int endfile;

    //! Size of each file
    std::vector<uint64_t> file_bytes;

    //! Total number of bytes in file set
    uint64_t total_bytes;

  };
}

#endif // !defined(__PdevFile_h)
