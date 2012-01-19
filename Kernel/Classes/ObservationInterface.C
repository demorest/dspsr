/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ObservationInterface.h"

dsp::Observation::Interface::Interface( Observation *c )
{
  add( &Observation::get_ndat,  "ndat",  "Number of time samples" );
  add( &Observation::get_nchan, "nchan", "Number of frequency channels" );
  add( &Observation::get_npol,  "npol",  "Number of polarizations" );
  add( &Observation::get_ndim,  "ndim",  "Number of data dimensions" );

  add( &Observation::get_type,
       &Observation::set_type,
       "type", "Observation type" );

  add( &Observation::get_telescope,
       &Observation::set_telescope,
       "site", "Telescope name" );

  add( &Observation::get_source,
       &Observation::set_source,
       "name", "Source name" );

  add( &Observation::get_coordinates,
       &Observation::set_coordinates,
       "coord", "Source coordinates" );

  add( &Observation::get_centre_frequency,
       &Observation::set_centre_frequency,
       "freq", "Centre frequency (MHz)" );

  add( &Observation::get_bandwidth,
       &Observation::set_bandwidth,
       "bw", "Bandwidth (MHz)" );

  add( &Observation::get_dispersion_measure,
       &Observation::set_dispersion_measure,
       "dm", "Dispersion measure (pc/cm^3)" );

  add( &Observation::get_rotation_measure,
       &Observation::set_rotation_measure,
       "rm", "Rotation measure (rad/m^2)" );

  add( &Observation::get_scale,
       &Observation::set_scale,
       "scale", "Data units" );

  add( &Observation::get_state,
       &Observation::set_state,
       "state", "Data state" );
  
  if (c)
    set_instance (c);
}

//! Set the instance
void dsp::Observation::Interface::set_instance (dsp::Observation* c) 
{
  TextInterface::To<Observation>::set_instance (c);
}

TextInterface::Parser *dsp::Observation::Interface::clone()
{
  if( instance )
    return new Interface( instance );
  else
    return new Interface();
}

