/*

 */

#ifndef __dsp_CASPSRUnpacker_h
#define __dsp_CASPSRUnpacker_h

#include "dsp/HistUnpacker.h"
//#include "dsp/CASPSRUnpacker.h"
namespace dsp {
  
  //
  class BitTable;
  
  class CASPSRUnpacker : public HistUnpacker
  {
  public:

    //! Constructor
    CASPSRUnpacker (const char* name = "CASPSRUpacker");

    //! Destructor
    virtual ~CASPSRUnpacker ();

    double get_optimal_variance ();

    void set_table (BitTable* table);

    const BitTable* get_table () const;


    void unpack (uint64_t ndat,
			 const unsigned char* from, const unsigned nskip,
			 float* into, const unsigned fskip,
			 unsigned long* hist);
  protected:
    
    Reference::To<BitTable> table;

    //! Return true if we can convert the Observation
    virtual bool matches (const Observation* observation);

    virtual void unpack ();
  };
}

#endif
