

#ifndef __EmerlinUnpacker_h
#define __EmerlinUnpacker_h


#include "dsp/Unpacker.h"
#include "dsp/TimeSeries.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/EmerlinTwoBitTable.h"
#include "dsp/TwoBitTable.h"


namespace dsp {
    class EmerlinUnpacker : public Unpacker {

        public:
            EmerlinUnpacker (const char* name="EmerlinUnpacker");
            unsigned get_ndig() const;


        protected:
            void unpack();
            bool matches(const Observation* observation);

            void reserve();
            void set_output(TimeSeries* _output);
            int get_ndat_per_weight();


        private:
            dsp::EmerlinTwoBitTable bittable;
            WeightedTimeSeries* weighted_output;

    };

}

#endif




