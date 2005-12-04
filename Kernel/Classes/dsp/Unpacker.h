//-*-C++-*-

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Classes/dsp/Unpacker.h,v $
   $Revision: 1.13 $
   $Date: 2005/12/04 23:55:57 $
   $Author: hknight $ */


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
  class Unpacker : public Transformation <const BitSeries, TimeSeries> {

  public:
    
    //! Constructor
    Unpacker (const char* name = "Unpacker");
    
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

    //! Provide BitSeries::input attribute access to derived classes
    template<class T>
    T* get_Input () const {
      {
	T* ptr = dynamic_cast<T*>( get_input()->input );
	if( ptr )
	  return ptr;
      }

      HoleyFile* hf = dynamic_cast<HoleyFile*>(get_input()->input);
	
      if( hf ){
	T* ptr = dynamic_cast<T*>( hf->get_loader() );
	
	if( ptr )
	  return ptr;
      }

      MultiFile* mf = dynamic_cast<MultiFile*>(get_input()->input);
	
      if( mf ){
	T* ptr = dynamic_cast<T*>( mf->get_loader() );
	
	if( ptr )
	  return ptr;
      }

      throw Error (InvalidState, "Unpacker::get_Input()",
		   "Yo BitSeries::input is not of required type- it is of type '%s' this type='%s'.  Value='%p'.  Its name=%s. hf=%p mf=%p",
		   typeid(get_input()->input).name(),
		   typeid(T*).name(),get_input()->input,
		   get_input()->input->get_name().c_str(),
		   hf, mf);
      return 0;
    }


  };

}

#endif // !defined(__Unpacker_h)
