noinst_LTLIBRARIES = libfits.la

nobase_include_HEADERS = dsp/FITSUnpacker.h dsp/FITSFile.h \
			 dsp/GUPPIFITSUnpacker.h

libfits_la_SOURCES = FITSUnpacker.C GUPPIFITSUnpacker.C \
		     FITSFile.C fits_params.h

#############################################################################
#

libfits_la_LIBADD = @CFITSIO_LIBS@

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += @CFITSIO_CFLAGS@

