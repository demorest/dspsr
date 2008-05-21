
noinst_LTLIBRARIES = libClasses.la

nobase_include_HEADERS = \
dsp/ASCIIObservation.h          dsp/InputBuffering.h            dsp/Seekable.h \
dsp/BitSeries.h                 dsp/InputBufferingShare.h       dsp/SetBufferingPolicy.h \
dsp/DADAFile.h                  dsp/JA98Unpacker.h              \
dsp/BitTable.h                  dsp/MPIRoot.h                   dsp/SubByteTwoBitCorrection.h \
dsp/BitUnpacker.h               dsp/MPIServer.h                 dsp/SubByte_dig_unpack.h \
dsp/BlockFile.h                 dsp/MPITrans.h                  dsp/TestInput.h \
dsp/BufferingPolicy.h           dsp/MiniExtension.h             \
dsp/ChannelOrder.h              dsp/MiniPlan.h                  \
dsp/DataSeries.h                dsp/TimeSeries.h \
dsp/DataSource.h                dsp/MultiFile.h                 dsp/Transformation.h \
dsp/Digitizer.h                 dsp/TwoBitCorrection.h \
dsp/EightBitUnpacker.h          dsp/Observation.h               dsp/TwoBitMask.h \
dsp/File.h                      dsp/TwoBitTable.h \
dsp/Operation.h                 dsp/Unpacker.h \
dsp/FourBitUnpacker.h           dsp/OutputBuffering.h           dsp/WeightedTimeSeries.h \
dsp/HistUnpacker.h              dsp/dsp.h                       dsp/dspExtension.h \
dsp/IOManager.h                 \
dsp/Input.h                     dsp/Scratch.h

libClasses_la_SOURCES = \
ASCIIObservation.C              InputBufferingShare.C           \
BitSeries.C                     JA98Unpacker.C                  SubByteTwoBitCorrection.C \
DADAFile.C                      TestInput.C \
BitTable.C                      \
BitUnpacker.C                   \
BlockFile.C                     TimeSeries.C \
ChannelOrder.C                  Transformation.C \
DataSeries.C                    TwoBitCorrection.C \
Digitizer.C                     MultiFile.C                     TwoBitTable.C \
EightBitUnpacker.C              Unpacker.C \
File.C                          Observation.C                   Unpacker_create.C \
Operation.C                     WeightedTimeSeries.C \
FourBitUnpacker.C               OutputBuffering.C               dsp.C \
HistUnpacker.C                  PseudoFile.C                    dspExtension.C \
IOManager.C                     Scratch.C                       \
Input.C                         Seekable.C                      \
InputBuffering.C                SetBufferingPolicy.C

if HAVE_MPI
libClasses_la_SOURCES += MPIRoot.C MPITrans.C MPIServer.C mpi_Observation.C
endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include

