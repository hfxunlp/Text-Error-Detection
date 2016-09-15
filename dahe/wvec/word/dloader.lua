function loadObject(fname)
	local file=torch.DiskFile(fname)
	local objRd=file:readObject()
	file:close()
	return objRd
end

--[[function loadseq(fname)
	local file=io.open(fname)
	local num=file:read("*n")
	local rs={}
	while num do
		local tmpt={}
		for i=1,num do
			local vi=file:read("*n")
			table.insert(tmpt,vi)
		end
		table.insert(rs,tmpt)
		num=file:read("*n")
	end
	file:close()
	return rs
end]]

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

wvec=loadObject('datasrc/wvec.asc')
sizvec=wvec:size(2)

mword=loadTrain('datasrc/thd/train','i.asc',359)

nsam=#mword

eaddtrain=ieps*nsam
