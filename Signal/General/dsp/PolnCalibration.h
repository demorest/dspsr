//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2009 by Ravi Kumar
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_PolnCalibration_h
#define __dsp_PolnCalibration_h

#include "dsp/Response.h"
#include "Pulsar/CalibratorTypes.h"
#include "Pulsar/PolnCalibrator.h"

namespace dsp {


class PolnCalibration: public Response {

  public:
     
  
    // constructor
    PolnCalibration ( );

    void set_ndat (unsigned _ndat)
    { ndat = _ndat; } 
    
    void set_nchan (unsigned _nchan)
    { nchan = _nchan;}
 
    void set_database_filename (const std::string& name)
    { database_filename = name; }

    void set_type ( Pulsar::Calibrator::Type* _type)
    {  type = _type; }

  void match (const Observation* input, unsigned channels=0);
  void match (const Response*);

    

  protected:
   
    unsigned ndat;
  
    unsigned nchan;

    std::string database_filename;

    Reference::To<Pulsar::Calibrator::Type> type;

    Reference::To<Pulsar::PolnCalibrator> pcal;


  };
}

#endif
