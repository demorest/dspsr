
noinst_LTLIBRARIES = libClasses.la

nobase_include_HEADERS = environ.h ascii_header.h \
	dsp/ASCIIObservation.h dsp/Seekable.h dsp/BitSeries.h	     \
	dsp/InputBuffering.h dsp/InputBufferingShare.h		     \
	dsp/DADAFile.h dsp/BitTable.h dsp/MPIRoot.h		     \
	dsp/SubByteTwoBitCorrection.h dsp/BitUnpacker.h		     \
	dsp/MPIServer.h dsp/BlockFile.h				     \
	dsp/MPITrans.h dsp/TestInput.h dsp/BufferingPolicy.h	     \
	dsp/MiniExtension.h dsp/ChannelOrder.h dsp/MiniPlan.h	     \
	dsp/DataSeries.h dsp/TimeSeries.h dsp/DataSource.h	     \
	dsp/MultiFile.h dsp/Transformation.h dsp/Digitizer.h	     \
	dsp/TwoBitCorrection.h dsp/EightBitUnpacker.h		     \
	dsp/Observation.h dsp/TwoBitMask.h dsp/File.h		     \
	dsp/TwoBitTable.h dsp/Operation.h dsp/Unpacker.h	     \
	dsp/FourBitUnpacker.h					     \
	dsp/WeightedTimeSeries.h dsp/HistUnpacker.h dsp/dsp.h	     \
	dsp/dspExtension.h dsp/IOManager.h dsp/Input.h dsp/Scratch.h \
	dsp/ExcisionUnpacker.h dsp/UnpackerIterator.h		     \
	dsp/BlockIterator.h dsp/StepIterator.h dsp/TwoBitLookup.h    \
	dsp/TwoBitFour.h dsp/TwoBit1or2.h dsp/FourBitTwo.h	     \
	dsp/EightBitOne.h dsp/excision_unpack.h dsp/NLowLookup.h     \
	dsp/ObservationChange.h dsp/SignalPath.h		     \
	dsp/infodata.h dsp/PrestoObservation.h			     \
	dsp/OutputArchive.h dsp/CloneArchive.h dsp/HasInput.h	     \
	dsp/HasOutput.h dsp/Sink.h dsp/Multiplex.h \
	dsp/Memory.h debug.h

libClasses_la_SOURCES = ascii_header.c ASCIIObservation.C	    \
	InputBufferingShare.C BitSeries.C SubByteTwoBitCorrection.C \
	DADAFile.C TestInput.C BitTable.C BitUnpacker.C BlockFile.C \
	TimeSeries.C ChannelOrder.C DataSeries.C   \
	TwoBitCorrection.C Digitizer.C MultiFile.C TwoBitTable.C    \
	EightBitUnpacker.C Unpacker.C File.C Observation.C	    \
	Unpacker_create.C Operation.C WeightedTimeSeries.C	    \
	FourBitUnpacker.C dsp.C HistUnpacker.C			    \
	dspExtension.C IOManager.C Scratch.C Input.C Seekable.C	    \
	InputBuffering.C ExcisionUnpacker.C    \
	TwoBitLookup.C TwoBitFour.C TwoBit1or2.C NLowLookup.C	    \
	UnpackerIterator.C ObservationChange.C PrestoObservation.C  \
	CloneArchive.C SignalPath.C Multiplex.C Memory.C

if HAVE_MPI
libClasses_la_SOURCES += MPIRoot.C MPITrans.C MPIServer.C mpi_Observation.C
endif

check_PROGRAMS = test_BlockIterator test_environ
test_BlockIterator_SOURCES = test_BlockIterator.C

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

LDADD = libClasses.la

