#! /bin/csh

set SFUSER = `awk -F@ '{print $1}' CVS/Root`
set SFPATH = ${SFUSER},dspsr@web.sourceforge.net:htdocs/classes

set SFHTML = dspsr

echo "Installing $SFHTML as $SFUSER"

cd html

rsync -avz $SFHTML $SFPATH/

