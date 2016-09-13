#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

from HTMLParser import HTMLParser

class processor(HTMLParser):
	def __init__(self):
		HTMLParser.__init__(self)
		self.procreset()

	def handle_starttag(self,tag,attrs):
		if tag=="error":
			self._cache.append(attrs)
		elif tag=="doc":
			self._cache=[]
		self._tag=tag

	def handle_data(self,data):
		self._cache.append(data)

	def handle_endtag(self,tag):
		if tag=="doc":
			self._rs.append(self._cache)
		self._tag=""

	def procreset(self):
		self._cache=[]
		self._rs=[]
		self._tag=""

	def getrs(self):
		return self._rs

def ldata(fsrc):
	proc=processor()
	with open(fsrc) as frd:
		proc.feed(frd.read().decode("utf-8"))
	rs=proc.getrs()
	proc.close()
	return rs

def filterlist(lin):
	lrs=[]
	for lu in lin:
		if isinstance(lu,basestring):
			tmp=lu.strip()
		else:
			tmp=lu
		if tmp:
			lrs.append(tmp)
	return lrs

def portalunit(lin):
	lin=filterlist(lin)
	estr=[i for i in lin[0]]
	cstr=[i for i in lin[1]]
	mktag=lin[2:]
	etag=["C" for i in xrange(len(estr))]
	for eu in mktag:
		putag=eu[-1][-1]
		st=int(eu[0][-1])-1
		ed=int(eu[1][-1])
		for i in xrange(st,ed):
			try:
				etag[i]=putag
			except:
				print lin[0]
	ed=zip(estr,etag)
	ed=[i[0]+"/"+i[-1] for i in ed]
	ed="  ".join(ed)
	cd="/C  ".join(cstr)+"/C"
	return [ed,cd]

def portalst(lin):
	rs=[]
	for lu in lin:
		rs.extend(portalunit(lu))
	return rs

def portal(fsrc,frs):
	with open(frs,"w") as fwrt:
		fwrt.write("\n".join(portalst(ldata(fsrc))).encode("utf-8"))

if __name__=="__main__":
	portal("ssrc.txt","tsrc.txt")
