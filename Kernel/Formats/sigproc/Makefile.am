
noinst_LTLIBRARIES = libsigproc.la

nobase_include_HEADERS = dsp/SigProcObservation.h dsp/SigProcDigitizer.h \
	dsp/SigProcFile.h dsp/SigProcUnpacker.h dsp/SigProcOutputFile.h

libsigproc_la_SOURCES = filterbank_header.c read_header.c send_stuff.c \
	error_message.c strings_equal.c swap_bytes.c \
	filterbank.h header.h sigproc.h polyco.h epn.h version.h \
	SigProcObservation.C SigProcDigitizer.C \
	SigProcFile.C SigProcUnpacker.C SigProcOutputFile.C

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

