//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Ravi Kumar
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __PolnCalibrator_h
#define __PolnCalibrator_h

#include "dsp/Response.h"
#include "Pulsar/CalibratorTypes.h"


namespace dsp {


class PolnCalibration: public Response {

  public:
     
  
    // constructor
    PolnCalibration ( );
     
    void set_database_filename (const std::string& name)
    { database_filename = name; }

    void set_type ( Pulsar::Calibrator::Type* _type)
    {  type = _type; }

    virtual void match (const Observation* input, unsigned channels=0);

    

  protected:

    std::string database_filename;

    Reference::To<Pulsar::Calibrator::Type> type;


  };
}

#endif
