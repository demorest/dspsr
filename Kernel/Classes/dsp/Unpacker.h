//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Unpacker.h,v $
   $Revision: 1.21 $
   $Date: 2008/02/20 09:29:05 $
   $Author: straten $ */


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
  class Unpacker : public Transformation <BitSeries, TimeSeries> {

  public:
    
    //! Constructor
    Unpacker (const char* name = "Unpacker");
    
    //! Return a pointer to a new instance of the appropriate sub-class
    static Unpacker* create (const Observation* observation);

    //! Return true if the derived class can convert the Observation
    /*! Derived classes must define the conditions under which they can
      be used to parse the given data. */
    virtual bool matches (const Observation* observation) = 0;

    //! Copy the input attributes to the output
    void prepare ();

    // Declare friends with Registry entries
    friend class Registry::Entry<Unpacker>;

    //! Get the number of digitizers (histograms)
    virtual unsigned get_ndig () const = 0;

   protected:

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
    T* get_Input () {

      const Input* ii = get_input()->get_loader();
      Input* iii = (Input*)(ii);
      
      {
	T* ptr = dynamic_cast<T*>( iii );
	if( ptr )
	  return ptr;
      }
#if 0
      HoleyFile* hf = dynamic_cast<HoleyFile*>( iii );
	
      if( hf ){
	T* ptr = dynamic_cast<T*>( hf->get_loader() );
	
	if( ptr )
	  return ptr;
      }

      MultiFile* mf = dynamic_cast<MultiFile*>( iii );
	
      if( mf ){
	T* ptr = dynamic_cast<T*>( mf->get_loader() );
	
	if( ptr )
	  return ptr;
      }
#endif

      throw Error (InvalidState, "Unpacker::get_Input",
		   "BitSeries::input is not of required type");

    }


  };

}

#endif // !defined(__Unpacker_h)
