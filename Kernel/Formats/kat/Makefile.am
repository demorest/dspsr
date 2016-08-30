
noinst_LTLIBRARIES = libkat.la

nobase_include_HEADERS =  dsp/KAT7Unpacker.h dsp/MeerKATUnpacker.h

libkat_la_SOURCES = KAT7Unpacker.C MeerKATUnpacker.C

if HAVE_CUDA

nobase_include_HEADERS += dsp/KAT7UnpackerCUDA.h dsp/MeerKATUnpackerCUDA.h
libkat_la_SOURCES += KAT7UnpackerCUDA.cu MeerKATUnpackerCUDA.cu

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

AM_CPPFLAGS += @CUDA_CFLAGS@
