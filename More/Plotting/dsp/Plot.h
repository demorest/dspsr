/***************************************************************************
 *
 *   Copyright (C) 2009 by Jonathan Khoo
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __dsp_Plot_h
#define __dsp_Plot_h

#include "cpgplot.h"

#include "dsp/BitSeries.h"
#include "dsp/ExcisionUnpacker.h"
#include "dsp/IOManager.h"
#include <Reference.h>
#include "dsp/TimeSeries.h"

#include "dsp/SampleDelay.h"
#include "dsp/SampleDelayFunction.h"
#include "dsp/DedispersionSampleDelay.h"

#define rangeType std::pair<float, float>

using std::string;
using std::vector;

namespace dsp
{
  class Plot : public Reference::Able
  {
    public:
      Plot();
      ~Plot();
      static bool verbose;
      virtual void transform();
      virtual void prepare() = 0;
      virtual void plot() = 0;
      virtual void set_filename(const std::string& fname);
      virtual void set_pol(const unsigned _pol);
      virtual void set_x_range(const rangeType& range);
      virtual void set_y_range(const rangeType& range);
      virtual string get_plot_title() const;

      // Input ndat
      virtual void set_ndat(const unsigned _ndat);
      virtual unsigned get_ndat() const;

      virtual void set_last_seconds(const float _last_seconds,
          float buffer = 0);

      virtual float get_last_seconds();
      virtual void finalise();

      uint64_t time_as_sample(const double time, const uint64_t ndat);

      // User-defined dispersion measure.
      float get_dispersion_measure();
      void set_dispersion_measure(const float _dispersion_measure);

      void set_dedisperse(const bool _dedisperse);
      bool get_dedisperse();

      void set_write_summed_channels(const bool _set_write_summed_channels);
      bool get_write_summed_channels() const;

    protected:
      double get_duration(const MJD& start, const MJD& end);
      void dedisperse_data(Observation* info);

      Reference::To<dsp::TimeSeries> voltages;
       Reference::To<dsp::SampleDelay> sample_delay;
      string filename;
      unsigned nbit;
      unsigned pol;
      unsigned ndat;
      double duration;

      double bandwidth;
      double centre_frequency;

      rangeType x_range;
      rangeType y_range;

      // pol, chan, dat?
      vector<vector<vector<float> > > data;

      float last_seconds;
      float buffer;

      bool dedisperse;

      float dispersion_measure;
      bool write_summed_channels;

      dsp::Observation* info;

    private:
      // Initialise plot-specific various before doing the transformation.
      void init(IOManager* manager);
  };
}

#endif
