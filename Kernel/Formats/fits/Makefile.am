noinst_LTLIBRARIES = libfits.la

nobase_include_HEADERS = dsp/FITSUnpacker.h dsp/FITSFile.h

libfits_la_SOURCES = FITSUnpacker.C FITSFile.C fits_params.h

#############################################################################
#

libfits_la_LIBADD = @CFITSIO_LIBS@

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @CFITSIO_CFLAGS@

