
noinst_LTLIBRARIES = liblofar_dal.la

nobase_include_HEADERS =  dsp/LOFAR_DALFile.h dsp/LOFAR_DALUnpacker.h

liblofar_dal_la_SOURCES = LOFAR_DALFile.C LOFAR_DALUnpacker.C 

liblofar_dal_la_LIBADD = -ldal -lhdf5

#############################################################################

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += -I/Users/willem/Pulsar/Darwin/include/dal
