//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Seekable.h,v $
   $Revision: 1.4 $
   $Date: 2002/11/06 06:30:42 $
   $Author: hknight $ */


#ifndef __Seekable_h
#define __Seekable_h

#include "dsp/Input.h"

namespace dsp {

  //! Pure virtual base class of sources that can seek through data
  /*! 
    This class defines the common interface as well as some basic
    functionality relating to sources of Timeseries data that can seek
  */
  class Seekable : public Input 
  {
    
  public:
    
    //! Constructor
    Seekable () { init(); }
    
    //! Destructor
    virtual ~Seekable () { }
    
    //! End of data
    virtual bool eod();
    
    //! Reset the file pointers
    virtual void reset ();

  protected:
    
    //! Load next block of data into Chronoseries
    virtual void load_data (Chronoseries* data);
 
    //! Load data from device and return the number of bytes read.
    virtual int64 load_bytes (unsigned char* buffer, uint64 bytes) = 0;
    
    //! Seek to absolute position and return absolute position in bytes
    virtual int64 seek_bytes (uint64 bytes) = 0;
    
    //! end of data reached
    bool end_of_data;
    
    //! Current time sample
    uint64 current_sample;
    
    //! initialize variables
    void init();
  };

}

#endif // !defined(__Seekable_h)
