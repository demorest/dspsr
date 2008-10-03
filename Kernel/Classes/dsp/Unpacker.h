//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Unpacker.h,v $
   $Revision: 1.27 $
   $Date: 2008/10/03 01:16:06 $
   $Author: straten $ */


#ifndef __Unpacker_h
#define __Unpacker_h

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"

#include "Registry.h"

namespace dsp {

  class Input;

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
  class Unpacker : public Transformation <BitSeries, TimeSeries>
  {

  public:
    
    //! Constructor
    Unpacker (const char* name = "Unpacker");
    
    //! Return a pointer to a new instance of the appropriate sub-class
    static Unpacker* create (const Observation* observation);

    //! Return true if the unpacker support the specified output order
    virtual bool get_order_supported (TimeSeries::Order) const;

    //! Set the order of the dimensions in the output TimeSeries
    virtual void set_output_order (TimeSeries::Order);

    //! Return true if the derived class can convert the Observation
    /*! Derived classes must define the conditions under which they can
      be used to parse the given data. */
    virtual bool matches (const Observation* observation) = 0;

    //! Match the unpacker to the resolution
    virtual void match_resolution (const Input*) {}

    //! Return ndat_per_weight
    virtual unsigned get_resolution () const { return 0; }

    //! Copy the input attributes to the output
    virtual void prepare ();

    // Declare friends with Registry entries
    friend class Registry::Entry<Unpacker>;

    //! Iterator through the input BitSeries
    class Iterator;

    //! Return the iterator for the specified digitizer
    Iterator get_iterator (unsigned idig);

   protected:

    //! The order of the dimensions in the output TimeSeries
    TimeSeries::Order output_order;

    //! The operation unpacks n-bit into floating point TimeSeries
    virtual void transformation ();
    
    //! The unpacking routine
    /*! This method must unpack the data from the BitSeries Input into
      the TimeSeries output. */
    virtual void unpack () = 0;

    //! Derived classes may redefine this
    virtual void resize_output ();

    //! Specialize the Unpacker for the Observation
    virtual void match (const Observation* observation);

    //! List of registered sub-classes
    static Registry::List<Unpacker> registry;

    //! Provide BitSeries::input attribute access to derived classes
    template<class T>
    const T* get_Input ()
    {
      const Input* input = get_input()->get_loader();

      const T* ptr = dynamic_cast<const T*>( input );
      if( ptr )
	return ptr;

      throw Error (InvalidState, "Unpacker::get_Input",
		   "BitSeries::input is not of required type");
    }
  };

}

#endif // !defined(__Unpacker_h)
