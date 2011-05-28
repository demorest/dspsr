%module dspsr
%{
#define SWIG_FILE_WITH_INIT
#include "numpy/noprefix.h"

#include "Reference.h"
#include "dsp/Operation.h"
#include "dsp/Observation.h"
#include "dsp/DataSeries.h"
#include "dsp/IOManager.h"
#include "dsp/BitSeries.h"
#include "dsp/TimeSeries.h"
#include "dsp/Detection.h"
#include "dsp/Dedispersion.h"
#include "dsp/Response.h"
#include "dsp/Convolution.h"

%}

// Language independent exception handler
%include exception.i       

%include std_string.i
%include stdint.i

using namespace std;

%exception {
    try {
        $action
    } catch(Error& error) {
        // Deal with out-of-range errors
        if (error.get_code()==InvalidRange)
            SWIG_exception(SWIG_IndexError, error.get_message().c_str());
        else
            SWIG_exception(SWIG_RuntimeError,error.get_message().c_str());
    } catch(...) {
        SWIG_exception(SWIG_RuntimeError,"Unknown exception");
    }
}

%init %{
  import_array();
%}

// Declare functions that return a newly created object
// (Helps memory management)
//%newobject Pulsar::Archive::new_Archive;
//%newobject Pulsar::Archive::load;

// Track any pointers handed off to python with a global list
// of Reference::To objects.  Prevents the C++ routines from
// prematurely destroying objects by effectively making python
// variables act like Reference::To pointers.
%feature("ref")   Reference::Able "pointer_tracker_add($this);"
%feature("unref") Reference::Able "pointer_tracker_remove($this);"
%header %{
std::vector< Reference::To<Reference::Able> > _pointer_tracker;
void pointer_tracker_add(Reference::Able *ptr) {
    _pointer_tracker.push_back(ptr);
}
void pointer_tracker_remove(Reference::Able *ptr) {
    std::vector< Reference::To<Reference::Able> >::iterator it;
    for (it=_pointer_tracker.begin(); it<_pointer_tracker.end(); it++) 
        if ((*it).ptr() == ptr) {
            _pointer_tracker.erase(it);
            break;
        }
}
%}

// Non-wrapped stuff to ignore
%ignore dsp::IOManager::add_extensions(Extensions*);
%ignore dsp::IOManager::combine(const Operation*);
%ignore dsp::IOManager::set_scratch(Scratch*);
%ignore dsp::BitSeries::set_memory(Memory*);
%ignore dsp::Convolution::Convolution(const char *, Behaviour);
%ignore dsp::Detection::set_engine(Engine*);
%ignore dsp::Observation::verbose_nbytes(uint64_t) const;

// Return psrchive's Estimate class as a Python tuple
%typemap(out) Estimate<double> {
    PyTupleObject *res = (PyTupleObject *)PyTuple_New(2);
    PyTuple_SetItem((PyObject *)res, 0, PyFloat_FromDouble($1.get_value()));
    PyTuple_SetItem((PyObject *)res, 1, PyFloat_FromDouble($1.get_variance()));
    $result = (PyObject *)res;
}
%typemap(out) Estimate<float> = Estimate<double>;

// Return psrchive's MJD class as a Python double.
// NOTE this loses precision so may not be appropriate for all cases.
%typemap(out) MJD {
    $result = PyFloat_FromDouble($1.in_days());
}

// Convert various enums to/from string
%define %map_enum(TYPE)
%typemap(out) Signal:: ## TYPE {
    $result = PyString_FromString( TYPE ## 2string($1).c_str());
}
%typemap(in) Signal:: ## TYPE {
    try {
        $1 = Signal::string2 ## TYPE (PyString_AsString($input));
    } catch (Error &error) {
        SWIG_exception(SWIG_RuntimeError,error.get_message().c_str());
    } 
}
%enddef
%map_enum(State)
%map_enum(Basis)
%map_enum(Scale)
%map_enum(Source)

// Header files included here will be wrapped
%include "ReferenceAble.h"
%include "dsp/Operation.h"
%include "dsp/Observation.h"
%include "dsp/DataSeries.h"
%include "dsp/IOManager.h"
%include "dsp/BitSeries.h"
%include "dsp/TimeSeries.h"
// Detection::Engine is screwing this up...
//%include "dsp/Detection.h"
%include "dsp/Dedispersion.h"
%include "dsp/Response.h"
%include "dsp/Convolution.h"

// Python-specific extensions to the classes:
%extend dsp::TimeSeries
{
    // Return a numpy array view of the data.
    // This points to the data array, not a separate copy.
    PyObject *get_dat(unsigned ichan, unsigned ipol)
    {
        PyArrayObject *arr;
        float *ptr;
        npy_intp dims[2];
        
        dims[0] = self->get_ndat();
        dims[1] = self->get_ndim();
        ptr = self->get_datptr(ichan, ipol);
        arr = (PyArrayObject *)                                         \
            PyArray_SimpleNewFromData(2, dims, PyArray_FLOAT, (char *)ptr);
        if (arr == NULL) return NULL;
        return (PyObject *)arr;
    }

    // Get the frac MJD part of the start time
    double get_start_time_frac()
    {
        return self->get_start_time().fracday();
    }
}
