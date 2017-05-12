
#ifndef __EmerlinFile_h
#define __EmerlinFile_h


#include <inttypes.h>
#include "dsp/File.h"
#include "dsp/BlockFile.h"


namespace dsp {


    class EmerlinFile : public File {

        public:
            EmerlinFile(const char* filename=0, const char* headername=0);

            ~EmerlinFile();

            bool is_valid(const char* filename) const ;

        protected:
            virtual void open_file(const char* filename);

            virtual int64_t seek_bytes(uint64_t bytes);
            virtual int64_t load_bytes(unsigned char* buffer, uint64_t nbytes);

        private:
            char datafile[1024];
            uint64_t cur_frame;
            uint64_t first_second;
            uint64_t dropped;

    };
}


#endif

