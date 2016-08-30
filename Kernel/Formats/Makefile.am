
noinst_LTLIBRARIES = libFormats.la

libFormats_la_SOURCES = File_registry.C Unpacker_registry.C

#############################################################################
#

include $(top_srcdir)/config/Makefile.bootstrap

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @PSRDADA_CFLAGS@ @CFITSIO_CFLAGS@ @GUPPI_DAQ_CFLAGS@ @CUDA_CFLAGS@

