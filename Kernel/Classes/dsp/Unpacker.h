//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Unpacker.h,v $
   $Revision: 1.8 $
   $Date: 2004/10/25 02:30:36 $
   $Author: hknight $ */


#ifndef __Unpacker_h
#define __Unpacker_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"

#include "Registry.h"

namespace dsp {

  //! Abstract base class of Transformations that convert n-bit to float
  /*! This class is used in conjunction with the File class in
    order to add new file formats to the baseband/dsp library.
    Inherit either dsp::Unpacker or one of its derived classes and
    implement the two pure virtual methods:

    <UL>
    <LI> bool matches(const Observation* observation)
    <LI> void unpack()
    </UL>

    then register the new class in Unpacker_registry.C.
  */
  class Unpacker : public Transformation <const BitSeries, TimeSeries> {

  public:
    
    //! Constructor
    Unpacker (const char* name = "Unpacker") 
      : Transformation <const BitSeries, TimeSeries> (name, outofplace,true) { }
    
    //! Return a pointer to a new instance of the appropriate sub-class
    static Unpacker* create (const Observation* observation);

    // Declare friends with Registry entries
    friend class Registry::Entry<Unpacker>;


   protected:
    //! The operation unpacks n-bit into floating point TimeSeries
    virtual void transformation ();
    
    //! The unpacking routine
    /*! This method must unpack the data from the BitSeries Input into
      the TimeSeries output. */
    virtual void unpack () = 0;

    //! Return true if the derived class can convert the Observation
    /*! Derived classes must define the conditions under which they can
      be used to parse the given data. */
    virtual bool matches (const Observation* observation) = 0;

    //! Specialize the Unpacker for the Observation
    virtual void match (const Observation* observation);

    //! List of registered sub-classes
    static Registry::List<Unpacker> registry;


  };

}

#endif // !defined(__Unpacker_h)
