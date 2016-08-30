
noinst_LTLIBRARIES = libcaspsr.la

nobase_include_HEADERS =  dsp/CASPSRUnpacker.h

libcaspsr_la_SOURCES = CASPSRUnpacker.C

if HAVE_CUDA

nobase_include_HEADERS += dsp/CASPSRUnpackerCUDA.h
libcaspsr_la_SOURCES += CASPSRUnpackerCUDA.cu

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

AM_CPPFLAGS += @CUDA_CFLAGS@
