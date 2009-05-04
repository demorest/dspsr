//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_dspExtension_h
#define __dsp_dspExtension_h

#include "Reference.h"
#include <string>

namespace dsp
{
  //! Extensions that can be added to the Extensions class
  class dspExtension : public Reference::Able
  {

  public:

    //! Constructor
    dspExtension (const std::string& name);

    //! Return a new copy-constructed instance identical to this instance
    virtual dspExtension* clone() const = 0;

    //! Delete this if dspExtension inherits from Printable
    std::string get_name() const { return name; }

  private:

    //! Delete this if dspExtension inherits from Printable
    std::string name;

  };


  class Extensions : public Reference::Able
  {
    public:

    //! Returns a pointer to the dspExtension
    template<class ExtensionType> ExtensionType* get();

    //! Returns a const pointer to the dspExtension
    template<class ExtensionType> const ExtensionType* get() const;

    //! Returns a pointer to the dspExtension
    //! If the dspExtension is not stored, adds a new null-constructed one
    template<class ExtensionType> ExtensionType* getadd();

    //! Adds a dspExtension
    void add_extension (dspExtension*);

    //! Returns the number of dspExtensions currently stored
    unsigned get_nextension () const;

    //! Returns the i'th dspExtension stored
    dspExtension* get_extension (unsigned iext);

    //! Returns the i'th dspExtension stored
    const dspExtension* get_extension (unsigned iext) const;

    protected:

    //! The vector of Extensions
    std::vector<Reference::To<dspExtension> > extension;

  };

  //! Returns a pointer to the dspExtension
  //! If the dspExtension is not stored this throws an Error
  template<class T>
  T* dsp::Extensions::get()
  {
    for( unsigned i=0; i<extension.size(); i++){
      T* ret = dynamic_cast<T*>(extension[i].get());
      if( ret )
        return ret;
    }

    return 0;
  }
  
  template<class T>
  const T* dsp::Extensions::get() const
  {
    for( unsigned i=0; i<extension.size(); i++){
      const T* ret = dynamic_cast<const T*>(extension[i].get());
      if( ret )
        return ret;
    }

    return 0;
  }

  
  //! Returns a pointer to the dspExtension
  //! If the dspExtension is not stored this adds a new null-instatntiated one
  template<class T>
  T* dsp::Extensions::getadd()
  {
    T* ext = get<T>();

    if (!ext)
    {
      ext = new T;  
      add_extension (ext);
    }

    return ext;
  }

}

#endif
