#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

def parm(strin):
	tmp=strin.split("\"")
	return (int(tmp[1]),int(tmp[3]))

def gm(cl):
	return gid(cl[0])

def gid(cu):
	id=str(cu[0])
	return "<ERROR start_off=\""+id+"\" end_off=\""+id+"\" type=\"M\"></ERROR>"

def merge(fsrc,frs):
	cache=[]
	with open(frs,"w") as fwrt:
		with open(fsrc) as frd:
			for line in frd:
				tmp=line.strip()
				if tmp:
					tmp=tmp.decode("utf-8")
					if tmp.endswith("type=\"M\"></ERROR>"):
						cumd=parm(tmp)
						if cache:
							if cache[-1][-1]+1==cumd[-1]:
								cache.append(cumd)
							else:
								stw=gm(cache)+"\n"
								fwrt.write(stw.encode("utf-8"))
								cache=[]
								stw=gid(cumd)+"\n"
								fwrt.write(stw.encode("utf-8"))
						else:
							cache.append(cumd)
					else:
						if cache:
							stw=gm(cache)+"\n"
							fwrt.write(stw.encode("utf-8"))
							cache=[]
						stw=tmp+"\n"
						fwrt.write(stw.encode("utf-8"))

if __name__=="__main__":
	merge("src.txt","ssrc.txt")
