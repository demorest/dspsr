//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2002 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Unpacker.h,v $
   $Revision: 1.17 $
   $Date: 2007/05/30 07:35:37 $
   $Author: straten $ */


#ifndef __Unpacker_h
#define __Unpacker_h

#include <typeinfo>

namespace dsp {
  class Unpacker;
}

#include "dsp/Transformation.h"
#include "dsp/TimeSeries.h"
#include "dsp/BitSeries.h"
#include "dsp/HoleyFile.h"
#include "dsp/MultiFile.h"

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

    // Declare friends with Registry entries
    friend class Registry::Entry<Unpacker>;

   protected:

    //! The operation unpacks n-bit into floating point TimeSeries
    virtual void transformation ();
    
    //! The unpacking routine
    /*! This method must unpack the data from the BitSeries Input into
      the TimeSeries output. */
    virtual void unpack () = 0;

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

      throw Error (InvalidState, "Unpacker::get_Input()",
		   "Yo BitSeries::input is not of required type- it is of type '%s' this type='%s'.  Value='%p'.  Its name=%s. hf=%p mf=%p",
		   typeid(iii).name(),
		   typeid(T*).name(),iii,
		   iii->get_name().c_str(),
		   hf, mf);
      return 0;
    }


  };

}

#endif // !defined(__Unpacker_h)
