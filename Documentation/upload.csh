#! /bin/csh

set SFUSER = `awk -F@ '{print $1}' CVS/Root`
set SFPATH = shell.sourceforge.net:/home/groups/d/ds/dspsr/htdocs/classes

set SFHTML = dspsr

echo "Installing $SFHTML as $SFUSER"

cd html
chmod -R g+w $SFHTML

rsync -Crvz --rsh="ssh -l $SFUSER" $SFHTML $SFPATH/

