#include "dsp/PhaseSeriesUnloader.h"
#include "dsp/PhaseSeries.h"
#include "polyco.h"
#include "string_utils.h"

string dsp::PhaseSeriesUnloader::get_filename (const PhaseSeries* data) const
{
  string filename = data->get_default_id ();

  if (data->get_integration_length() < 1.0) {

    // small files will need unique filenames

    const polyco* poly = data->get_folding_polyco();
    if (poly) {
      // add pulse number to the output archive
      Phase phase = poly->phase ( data->get_start_time() );
      phase = (phase + 0.5-data->get_reference_phase()).Ceil();

      filename += stringprintf ("."I64, phase.intturns());
    }
      
  }

  filename += ".ar";

  return filename;
}

//! Set the PhaseSeries from which Profile data will be constructed
void dsp::PhaseSeriesUnloader::set_profiles (const PhaseSeries* _profiles)
{
  profiles = _profiles;
}
