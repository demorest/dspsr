//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2004 - 2011 by Haydon Knight and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_BCPMFile_h
#define __dsp_BCPMFile_h

#include "dsp/File.h"

namespace dsp
{
  //! Loads BitSeries data from a BCPM data file.
  class BCPMFile : public File 
  {

  public:
   
    //! Construct and open file
    BCPMFile (const char* filename = 0);

    //! Returns true if filename appears to name a valid BCPM file
    bool is_valid (const char* filename) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Pulled out of sigproc
    void bpp_chans (std::vector<int>& table,
		    double bw, int mb_start_addr, int mb_end_addr, int mb_start_brd, int mb_end_brd, int *cb_id, double *aib_los, float *dfb_sram_freqs, double rf_lo,
		    double& centre_frequency,
		    double& channel_bandwidth);

    //! Pads gaps in data.  Untested
    virtual int64_t pad_bytes(unsigned char* buffer, int64_t bytes);

    friend class BCPMUnpacker;

    //! This stores the ordering of the channels in a BCPM data file
    std::vector<int> chtab;
  };

}

#endif // !defined(__BCPMFile_h)
  
