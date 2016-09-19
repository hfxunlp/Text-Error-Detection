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
		tart=torch.Tensor(cld:size(2)):fill(1)
		tar={}
		for _i=1,cld:size(1) do
			table.insert(tar,tart)
		end
		table.insert(rtar,tar)
	end
	return id,rtar
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

function glmodi(modin)
	local rt=nil
	local tidf=torch.zeros(2,1)
	if modin then
		rt={tidf,modin:forward(tidf)}
	else
		rt=tidf
	end
	return rt
end