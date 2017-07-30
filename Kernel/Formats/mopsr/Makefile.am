
noinst_LTLIBRARIES = libmopsr.la

nobase_include_HEADERS =  dsp/MOPSRUnpacker.h

libmopsr_la_SOURCES = MOPSRUnpacker.C

include $(top_srcdir)/config/Makefile.include

if HAVE_CUDA

nobase_include_HEADERS += dsp/MOPSRUnpackerCUDA.h
libmopsr_la_SOURCES += MOPSRUnpackerCUDA.cu

include $(top_srcdir)/config/Makefile.cuda
endif


AM_CPPFLAGS += @CUDA_CFLAGS@
