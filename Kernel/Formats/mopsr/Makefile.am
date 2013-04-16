
noinst_LTLIBRARIES = libmopsr.la

nobase_include_HEADERS =  dsp/MOPSRUnpacker.h

libmopsr_la_SOURCES = MOPSRUnpacker.C

if HAVE_CUDA

nobase_include_HEADERS += dsp/MOPSRUnpackerCUDA.h
libmopsr_la_SOURCES += MOPSRUnpackerCUDA.cu

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @CUDA_CFLAGS@
