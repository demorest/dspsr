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

 cerr << "MJD=" << subint->get_epoch() << endl;
 cerr << "bw=" << archive->get_bandwidth() << endl;
 cerr << "cf=" << archive->get_centre_frequency() << endl;
 cerr << "coord=" << archive->get_coordinates() << endl;
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

    unsigned nchan = 128;

    // 1 sub-integration, 4 polarizations, nchan channels
    archive->resize (1, 4, nchan);

    // the following line is equivalent to
    // Pulsar::Database* dbase = 0;
    Reference::To<Pulsar::Database> dbase;

    dbase = new Pulsar::Database (database_filename);

    Pulsar::Database::Criterion::match_verbose = true;
    Pulsar::PolnCalibrator::verbose = true;
    Pulsar::Database::verbose = true;

    // default searching criterion
    Pulsar::Database::Criterion criterion;
    criterion.check_coordinates = false;
    Pulsar::Database::set_default_criterion (criterion);

    Reference::To<Pulsar::PolnCalibrator> pcal;

    type = new Pulsar::CalibratorTypes::SingleAxis;


    pcal = dbase->generatePolnCalibrator (archive , type);

    cerr << "PCAL LOADED nchan=" << pcal->get_nchan() << endl;

    // pcal->set_response_nchan (128);

    vector < Jones<float> > R (nchan);
     // unsigned nchan = pcal->get_nchan();
 
    for ( unsigned ichan = 0 ; ichan < channels ; ichan++)
      R[ichan] = pcal -> get_response (ichan);

    set (R);

    cerr << "PolnCalibration::match finished!" << endl;
}
