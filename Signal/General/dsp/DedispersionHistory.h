//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2005 by Haydon Knight
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_DedispersionHistory_h_
#define __dsp_DedispersionHistory_h_

#include "dsp/dspExtension.h"

namespace dsp {

  class DedispersionHistory : public dspExtension {
	
  public:

    //! Null constructor
    DedispersionHistory();

    //! Virtual destructor
    virtual ~DedispersionHistory();

    //! Return a new copy-constructed instance identical to this instance
    virtual dspExtension* clone() const;
	
    //! Add in a dedispersion operation
    void add(std::string classname, float dm);

    std::vector<float> get_dms(){ return dms; }

  private:

    //! The classes that did the dedispersion
    std::vector<std::string> classes;

    //! The DMs used
    std::vector<float> dms;
  };

}

#endif
