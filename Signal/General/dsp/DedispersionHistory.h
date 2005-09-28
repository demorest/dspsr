//-*-C++-*-

#ifndef __dsp_DedispersionHistory_h_
#define __dsp_DedispersionHistory_h_

#include <vector>
#include <string>

#include "psr_cpp.h"

#include "dsp/dspExtension.h"

namespace dsp {

    class DedispersionHistory : public dspExtension {
	
    public:

	//! Null constructor
	DedispersionHistory();

	//! Virtual destructor
	virtual ~DedispersionHistory();

	//! Return a new copy-constructed instance identical to this instance
	virtual dspExtension* clone() const;
	
	//! Return a new null-constructed instance
	virtual dspExtension* new_extension() const;

        //! Dump out to a string
        virtual string dump_string() const;

	//! Add in a dedispersion operation
	void add(string classname, float dm);

	vector<float> get_dms(){ return dms; }

    private:

	//! The classes that did the dedispersion
	vector<string> classes;

	//! The DMs used
	vector<float> dms;
    };

}

#endif
