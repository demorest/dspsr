
## This is a simple example of how to use the dspsr
## python interface to load raw data.

import dspsr

# Set up a TimeSeries object, and IOManager:
data = dspsr.TimeSeries()
loader = dspsr.IOManager()

# Connect the output of the IOManager to the TimeSeries:
loader.set_output(data)

# Open the data file:
#loader.open('my_raw_datafile.dat')
loader.open('TPUL0001_Lband_raw.57324.89902427084.4774.B1937+21.AC-00.0000.raw')

# Set the load block size in samples:
loader.set_block_size(16*1024)

# Each time loader.operate() is run, a new block of data 
# is loaded in:
loader.operate()

# Once some data have been loaded, data.get_data() returns
# a numpy array view of the data array.  The arguments
# are the channel index and polarization index:
print "Data have dimensions", data.get_dat(0,0).shape

# loader.operate() will return False when there is no
# more data.  To iterate over all data in a file, do
# something like:
while loader.operate():
    print "The max value for this block is", \
            data.get_dat(0,0).max()
