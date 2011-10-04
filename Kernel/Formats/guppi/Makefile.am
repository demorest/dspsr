
noinst_LTLIBRARIES = libguppi.la

nobase_include_HEADERS =  dsp/GUPPIFile.h \
			  dsp/GUPPIBlockFile.h \
			  dsp/GUPPIUnpacker.h 

libguppi_la_SOURCES = GUPPIFile.C \
		      GUPPIBlockFile.C \
		      GUPPIUnpacker.C \
		      fitshead_utils.C hget.c 

if HAVE_GUPPI_DAQ

nobase_include_HEADERS += dsp/GUPPIBuffer.h
libguppi_la_SOURCES += GUPPIBuffer.C

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

LDADD = @GUPPI_DAQ_LIBS@
AM_CPPFLAGS += @GUPPI_DAQ_CFLAGS@
