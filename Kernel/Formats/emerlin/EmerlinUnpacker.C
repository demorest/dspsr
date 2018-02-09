
#include "dsp/EmerlinUnpacker.h"
#include "dsp/WeightedTimeSeries.h"


dsp::EmerlinUnpacker::EmerlinUnpacker(const char* name) : Unpacker(name) {
}

bool dsp::EmerlinUnpacker::matches (const Observation* observation) {

    return observation->get_machine() == "EMERLIN"
        && observation->get_nbit() == 2
        && (observation->get_npol() == 2 || observation->get_npol() == 1);

}

int dsp::EmerlinUnpacker::get_ndat_per_weight() {
    return 32000; 
}


void dsp::EmerlinUnpacker::reserve() {
    if (weighted_output)
    {
        weighted_output -> set_ndat_per_weight (get_ndat_per_weight());
        weighted_output -> set_nchan_weight (1);
        weighted_output -> set_npol_weight (input->get_npol());
    }

    output->resize ( input->get_ndat() );

    if (weighted_output)
        weighted_output -> neutral_weights ();
}

void dsp::EmerlinUnpacker::set_output (TimeSeries* _output)
{
    if (verbose)
        std::cerr << "dsp::EmerlinUnpacker::set_output (" << _output << ")" << std::endl;

    Unpacker::set_output (_output);
    weighted_output = dynamic_cast<WeightedTimeSeries*> (_output);
}

void dsp::EmerlinUnpacker::unpack() {
    if(verbose) {
        std::cerr << "dsp::EmerlinUnpacker::unpack()" << std::endl;
        std::cerr << "dsp::EmerlinUnpacker input->ndat = "<< input->get_ndat() << std::endl;
        std::cerr << "dsp::EmerlinUnpacker input->nbit = "<< input->get_nbit() << std::endl;
        std::cerr << "dsp::EmerlinUnpacker input->ndim = "<< input->get_ndim() << std::endl;
        std::cerr << "dsp::EmerlinUnpacker input->npol = "<< input->get_npol() << std::endl;
        std::cerr << "dsp::EmerlinUnpacker input->nchan = "<< input->get_nchan() << std::endl;
        std::cerr << "dsp::EmerlinUnpacker output->ndat = "<< output->get_ndat() << std::endl;
    }


    const unsigned samples_per_byte=4;

    const unsigned total_bytes = input->get_npol()*input->get_ndat()/samples_per_byte;

    const unsigned nframe = total_bytes/(input->get_npol()*8000);
    const unsigned nword = 2000;
    const unsigned byte_per_word=4;
    unsigned offset=0;

    const unsigned dat_per_frame = nword*byte_per_word*samples_per_byte;
    unsigned weights_per_frame = 0;
    if(weighted_output){
        weights_per_frame = dat_per_frame / weighted_output->get_ndat_per_weight();
        if(verbose)
            std::cerr << "dsp::EmerlinUnpacker weighted output. weights per frame = " << weights_per_frame << std::endl;
    }

    const unsigned char *iarray = input->get_rawptr();
    const unsigned char *iarray_orig = iarray;
    unsigned char word[byte_per_word];

    int count[4];

    unsigned* weights = NULL;
    for (unsigned iframe=0; iframe < nframe; ++iframe) {
        for (unsigned ipol=0; ipol < input->get_npol(); ++ipol) {
            if(weighted_output){
                if(verbose) {
                    std::cerr << "dsp::EmerlinUnpacker get weights ipol="<< ipol<<std::endl;
                }
                weights = weighted_output->get_weights(0,ipol)+weights_per_frame*iframe;
            }
            count[0]=0;
            count[1]=0;
            count[2]=0;
            count[3]=0;
            float ss=0;
            if(offset > output->get_ndat()){
                std::cerr << "dsp::EmerlinUnpacker::unpack error" << std::endl;
            }

            float* oarray = output->get_datptr (0, ipol) + offset;
            for (unsigned wd=0; wd < nword; ++wd) {
                for (unsigned bt = 0; bt < byte_per_word; bt++){
                    //                    word[bt] = iarray[byte_per_word-1-bt]; // first samples are in last byte of word.
                    word[bt] = iarray[bt]; // first sample is byte zero on disk.
                }

                iarray += 4;


                for (unsigned bt = 0; bt < byte_per_word; bt++){
                    const float* four = bittable.get_values(word[bt]);
                    //    std::cerr << (int)(word[bt]) << std::endl;
                    //    std::cerr << four[0] << " " << four[1] <<
                    //        " " << four[2] << " " << four[3] << std::endl;

                    for (unsigned pt=0; pt < samples_per_byte; ++pt) {
                        if (four[pt] < -0.5)count[0]++;
                        else if(four[pt] < 0)count[1]++;
                        else if(four[pt] < 0.5) count[2]++;
                        else count[3]++;
                        *oarray = four[pt];
                        ss+=four[pt]*four[pt];
                        ++oarray;
                    }
                }
            }
            if(count[3]==0 && count[2]==0 && count[1]==0){
                std::cerr << "Zero weight Dropped Frame (weights_per_frame="<<weights_per_frame<<")" << std::endl;
                std::cerr << -bittable.get_hi_val() << " : " << count[0] << std::endl;
                std::cerr << -bittable.get_lo_val() << " : " << count[1] << std::endl;
                std::cerr << bittable.get_lo_val() << " : " << count[2] << std::endl;
                std::cerr << bittable.get_hi_val() << " : " << count[3] << std::endl;

                for (int iw=0; iw < weights_per_frame; ++iw) {
                    weights[iw] = 0;
                }
            }
            if(verbose){
                std::cerr << "dsp::EmerlinUnpacker::unpack frame=" <<iframe<<" pol="<<ipol << std::endl;
                std::cerr << -bittable.get_hi_val() << " : " << count[0] << std::endl;
                std::cerr << -bittable.get_lo_val() << " : " << count[1] << std::endl;
                std::cerr << bittable.get_lo_val() << " : " << count[2] << std::endl;
                std::cerr << bittable.get_hi_val() << " : " << count[3] << std::endl;
                std::cerr << "SSSS" << ipol << " " << ss << std::endl;
            }


        }

        offset += 8000*samples_per_byte;
    }

}
