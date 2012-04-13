#include "dsp/Plot.h"

typedef std::map<float, unsigned> mapType;
typedef std::pair<float, unsigned> pairType;

namespace dsp
{
    class HistoPlot : public Plot
    {
        public:
            HistoPlot();
            ~HistoPlot();
            virtual void prepare();
            virtual void plot();

        protected:
            mapType hist;
            float getBinInterval();
            bool allSamplesUsed();
            float getMaxValue();
    };
}

bool getMax(pairType p1, pairType p2);
std::pair<float, float> get_mean_rms(mapType& hist);
