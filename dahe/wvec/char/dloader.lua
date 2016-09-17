function loadObject(fname)
	local file=torch.DiskFile(fname)
	local objRd=file:readObject()
	file:close()
	return objRd
end

function loadSeqTensor(fname)
	local file=io.open(fname)
	local lind=file:read("*n")
	local rs={}
	local num=file:read("*n")
	while num do
		local tmpt={}
		for i=1,lind do
			table.insert(tmpt,num)
			num=file:read("*n")
		end
		table.insert(rs,torch.Tensor(tmpt))
	end
	file:close()
	return rs
end

function loadTrain(iprefix,ifafix,nfile)
	local id={}
	for i=1,nfile do
		table.insert(id,loadObject(iprefix..i..ifafix))
	end
	return id
end

function loadnt(iprefix,ifafix,nfile,ntotal)
	local id={}
	local rtar={}
	local curld=nil
	local cld=nil
	local culen=nil
	local tart=nil
	local tar=nil
	for i=1,nfile do
		curld=math.random(ntotal)
		while colid[curld] do
			curld=math.random(ntotal)
		end
		colid[curld]=true
		table.insert(colidx,curld)
		cld=loadObject(iprefix..curld..ifafix)
		table.insert(id,cld)
		culen=cld:size(1)-1
		tart=torch.Tensor(cld:size(2)):fill(1)
		tar={}
		for _i=1,cld:size(1)-1 do
			table.insert(tar,tart)
		end
		table.insert(rtar,tar)
	end
	return id,rtar
end

function rldc()
	local apin
	local aptar
	apin,aptar=loadnt('datasrc/thd/train','i.asc',nfresh,nsam)
	for _tmpi=1,nfresh do
		table.remove(mword,1)
		table.insert(mword,table.remove(apin))
		table.remove(mwordt,1)
		table.insert(mwordt,table.remove(aptar))
		colid[table.remove(colidx,1)]=nil
	end
end

function rldt()
	rldc()
	collectgarbage()
end

function glmodi()
	local ti=mword[1]
	local tif=ti:narrow(1,1,1)
	local til=ti:narrow(1,2,1)
	tif=tif:narrow(2,1,1):clone()
	til=til:narrow(2,1,1):clone()
	return {tif:zero(),til:zero()}
end

wvec=loadObject('datasrc/wvec.asc')
sizvec=wvec:size(2)

colid={}
colidx={}
mword,mwordt=loadnt('datasrc/thd/train','i.asc',tld,nsam)
lmodi=glmodi()