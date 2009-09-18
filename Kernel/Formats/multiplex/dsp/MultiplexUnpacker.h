/*

 */

#ifndef __dsp_MultiplexUnpacker_h
#define __dsp_MultiplexUnpacker_h

#include "dsp/HistUnpacker.h"
//#include "dsp/MultiplexUnpacker.h"
namespace dsp {
  
  //
  class BitTable;
  
  class MultiplexUnpacker : public HistUnpacker
  {
  public:

    //! Constructor
    MultiplexUnpacker (const char* name = "MultiplexUpacker");

    //! Destructor
    virtual ~MultiplexUnpacker ();

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
