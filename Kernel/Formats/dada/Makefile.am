
noinst_LTLIBRARIES = libdada.la

nobase_include_HEADERS =  dsp/DADABuffer.h

libdada_la_SOURCES =  DADABuffer.C

libdada_la_LIBADD = @PSRDADA_LIBS@

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @PSRDADA_CFLAGS@ @CUDA_CFLAGS@

