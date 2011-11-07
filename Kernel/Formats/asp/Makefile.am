
noinst_LTLIBRARIES = libasp.la

nobase_include_HEADERS =  dsp/ASPFile.h dsp/ASPUnpacker.h 

libasp_la_SOURCES = ASPFile.C ASPUnpacker.C asp_params.h data2rcv.h

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

# This is a bit of a hack since ASPUnpacker is used for both 
# asp and pdev data, and uses the PdevFile class.
AM_CPPFLAGS += -I$(top_srcdir)/Kernel/Formats/pdev
