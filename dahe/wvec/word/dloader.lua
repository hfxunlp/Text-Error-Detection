require "ldapi"

wvec=loadObject('datasrc/wvec.asc')
sizvec=wvec:size(2)

colid={}
colidx={}
mword=loadnt('datasrc/thd/train','i.asc',tld,nsam)