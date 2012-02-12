#include "dsp/Plot.h"

namespace dsp
{
  class FrequencyVsTime : public Plot
  {
    public:
      FrequencyVsTime();
      ~FrequencyVsTime();
      virtual void prepare();
      virtual void plot();
      virtual void finalise();

    protected:
      void preparePlotScale();
      void makeDedispersedChannelSums();
      void write_summed_channels_to_file();

      std::vector<float> values;

      std::vector<float> data_sums;

      unsigned ndat;
      unsigned nchan;
      unsigned npol;
  };
}

template <class T>
T sumOfVectorElements(vector<T>& v);
