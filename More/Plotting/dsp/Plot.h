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

      uint64_t time_as_sample(const double time, const uint64_t ndat, 
          const double duration);

      // User-defined dispersion measure.
      float get_dispersion_measure();
      void set_dispersion_measure(const float _dispersion_measure);

    protected:
      void dedisperse_data(Observation* info);

      Reference::To<dsp::TimeSeries> voltages;
      Reference::To<dsp::SampleDelay> sample_delay;

      Reference::To<Observation> info;

      string filename;
      unsigned pol;

      unsigned ndat;

      rangeType x_range;
      rangeType y_range;

      // pol, chan, dat?
      vector<vector<vector<float> > > data;

      float last_seconds;
      float buffer;

      bool dedisperse;

      float dispersion_measure;

    private:
      // Initialise plot-specific various before doing the transformation.
      void init(IOManager* manager);
  };
}

#endif
