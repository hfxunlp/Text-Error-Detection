torch.setdefaulttensortype('torch.LongTensor')

function loadObject(fname)
	local file=torch.DiskFile(fname)
	local objRd=file:readObject()
	file:close()
	return objRd
end

function convfile(fsrc,frs)
	local tsd=loadObject(fsrc):long()
	file=torch.DiskFile(frs,'w')
	file:writeObject(tsd)
	file:close()
end

for nf=1,178558 do
	convfile("thd/train"..nf.."i.asc","nd/train"..nf.."i.asc",true)
end