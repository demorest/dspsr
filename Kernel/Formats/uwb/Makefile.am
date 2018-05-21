
noinst_LTLIBRARIES = libuwb.la

nobase_include_HEADERS =  dsp/UWBUnpacker.h

libuwb_la_SOURCES = UWBUnpacker.C

if HAVE_CUDA

nobase_include_HEADERS += dsp/UWBUnpackerCUDA.h
libuwb_la_SOURCES += UWBUnpackerCUDA.cu

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

AM_CPPFLAGS += @CUDA_CFLAGS@
