
noinst_LTLIBRARIES = libska1.la

nobase_include_HEADERS =  dsp/SKA1Unpacker.h

libska1_la_SOURCES = SKA1Unpacker.C

if HAVE_CUDA

nobase_include_HEADERS += dsp/SKA1UnpackerCUDA.h
libska1_la_SOURCES += SKA1UnpackerCUDA.cu

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

AM_CPPFLAGS += @CUDA_CFLAGS@
