/***************************************************************************
 *
 *   Copyright (C) 2009 by Ravi Kumar
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/


#include "dsp/PolnCalibration.h"
#include <iostream>
#include <unistd.h>
#include <Jones.h>

#include "dsp/File.h"
#include "dsp/Response.h"

#include "Pulsar/BasicArchive.h"
#include "Pulsar/Integration.h"
#include "Pulsar/Database.h"
#include "Pulsar/CalibratorTypes.h"
#include "Pulsar/PolnCalibrator.h"
#include "Pulsar/Backend.h"
#include "Pulsar/Receiver.h"

using namespace std;


void cpy_attributes ( const dsp::Observation* obs , Pulsar::Archive* archive )
{ 

 archive->set_source ( obs->get_source() );

 archive->set_state ( obs->get_state() );

 archive->set_type ( obs->get_type() );

 archive->set_coordinates ( obs->get_coordinates() );

 archive->set_bandwidth ( obs->get_bandwidth() );

 archive->set_centre_frequency ( obs->get_centre_frequency() );

 archive->set_dispersion_measure  ( obs->get_dispersion_measure () );

 archive->set_rotation_measure  ( obs->get_rotation_measure () );

 Pulsar::Backend* backend = archive->getadd<Pulsar::Backend> ();
 backend->set_name( obs->get_machine() );

 Pulsar::Receiver* receiver = archive->getadd<Pulsar::Receiver> ();
 receiver->set_name( obs->get_receiver());

 archive->resize (1);
 Pulsar::Integration* subint = archive->get_Integration (0);
 subint->set_epoch( obs->get_start_time() );

 if (dsp::Operation::verbose)
 {
   cerr << "cpy_attributes MJD=" << subint->get_epoch() << endl;
   cerr << "cpy_attributes bw=" << archive->get_bandwidth() << endl;
   cerr << "cpy_attributes cf=" << archive->get_centre_frequency() << endl;
   cerr << "cpy_attributes coord=" << archive->get_coordinates() << endl;
 }

}


class StubArchive : public Pulsar::BasicArchive
{
public:

  bool can_unload() const
  { return false; }

  Pulsar::Archive* clone() const
  { return new StubArchive(*this); }

  void load_header (const char*)
  { throw Error (InvalidState, "StubArchive::load_header",
		 "not implemented"); }

  Pulsar::Integration*
  load_Integration (const char*, unsigned int)
  { throw Error (InvalidState, "StubArchive::load_Integration",
		 "not implemented"); }

  void unload_file(const char*) const
  { throw Error (InvalidState, "StubArchive::unload_file",
		 "not implemented"); }
};


dsp::PolnCalibration::PolnCalibration ()
{
}

void dsp::PolnCalibration::match (const Observation* input, unsigned channels)
{

    Reference::To<Pulsar::Archive> archive = new StubArchive;

    cpy_attributes (input , archive);

    // unsigned nchan = 128;

    // 1 sub-integration, 4 polarizations, ndat channels
    // archive->resize (1, 4, ndat);

    // the following line is equivalent to
    // Pulsar::Database* dbase = 0;
    Reference::To<Pulsar::Database> dbase;

    dbase = new Pulsar::Database (database_filename);

    // Pulsar::Database::Criterion::match_verbose = true;
    // Pulsar::PolnCalibrator::verbose = true;
    // Pulsar::Database::verbose = true;

    // default searching criterion
    Pulsar::Database::Criteria criteria;
    criteria.check_coordinates = false;
    Pulsar::Database::set_default_criteria (criteria);

    type = new Pulsar::CalibratorTypes::SingleAxis;


    pcal = dbase->generatePolnCalibrator (archive , type);
}

void dsp::PolnCalibration::match (const Response* boss)
{
  unsigned ndat = boss->get_ndat() * boss->get_nchan();

  if (verbose)
    cerr << "dsp::PolnCalibration::match WANT ndat=" << ndat << endl;

  pcal->set_response_nchan (ndat);

  if (verbose)
    cerr << "dsp::PolnCalibration::match PCAL LOADED nchan=" 
         << pcal->get_response_nchan() << endl;

  vector < Jones<float> > R (ndat);
 
  for ( unsigned ichan = 0 ; ichan < ndat; ichan++)
    R[ichan] = pcal -> get_response (ichan);

  set (R);
 
  if (verbose) 
    cerr << "dsp::PolnCalibration::match resize" <<endl;

  resize (1, boss->get_nchan(), boss->get_ndat(), 8);

}
