#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

def ldd(df):
	rsd={}
	with open(df) as frd:
		for line in frd:
			tmp=line.strip()
			if tmp:
				tmp=tmp.decode("utf-8")
				tmp=tmp.split("	")
				if len(tmp)==2:
					rsd[tmp[0]]=tmp[-1]
	return rsd

def mapf(srcf,rsf,mapd):
	with open(rsf,"w") as fwrt:
		with open(srcf) as frd:
			for line in frd:
				tmp=line.strip()
				if tmp:
					tmp=tmp.decode("utf-8")
					rsl=[]
					for tmpu in tmp:
						rsl.append(mapd.get(tmpu,"0"))
					tmp=str(len(rsl))+" "+" ".join(rsl)+"\n"
					fwrt.write(tmp)

if __name__=="__main__":
	mapf("tsrc.txt","tlua.txt",ldd("wd.txt"))