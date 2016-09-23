function loadObject(fname)
	local file=torch.DiskFile(fname)
	local objRd=file:readObject()
	file:close()
	return objRd
end

function loadnt(iprefix,ifafix,nfile,ntotal)
	local id={}
	local curld,cld,tart,tar=nil
	for i=1,nfile do
		curld=math.random(ntotal)
		while colid[curld] do
			curld=math.random(ntotal)
		end
		colid[curld]=true
		table.insert(colidx,curld)
		cld=loadObject(iprefix..curld..ifafix)
		table.insert(id,cld)
	end
	return id
end

function rldc()
	local apin,aptar=loadnt('datasrc/thd/train','i.asc',nfresh,nsam)
	for _tmpi=1,nfresh do
		table.remove(mword,1)
		table.insert(mword,table.remove(apin))
		table.remove(mwordt,1)
		table.insert(mwordt,table.remove(aptar))
		table.remove(colid,table.remove(colidx,1))
	end
end

function rldt()
	rldc()
	collectgarbage()
end