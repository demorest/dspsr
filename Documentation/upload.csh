#! /bin/csh

set SFUSER = `awk -F@ '{print $1}' CVS/Root`
set SFPATH = /home/groups/d/ds/dspsr/htdocs/classes

echo "Installing html/* as $SFUSER"

echo "Creating gzipped tarball ..."
cd html
chmod -R g+w dspsr
tar cf doc.tar dspsr
gzip -f doc.tar

echo "Secure copying to shell.sourceforge.net ..."

scp doc.tar.gz $SFUSER@shell.sourceforge.net:
ssh $SFUSER@shell.sourceforge.net "cd $SFPATH && rm -rf * && tar zxvf ~/doc.tar.gz"

