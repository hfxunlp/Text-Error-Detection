torch.setdefaulttensortype('torch.FloatTensor')

function convec(fsrc,frs,lsize)
	local file=io.open(fsrc)
	local num=file:read("*n")
	local rs={}
	while num do
		table.insert(rs,num)
		num=file:read("*n")
	end
	file:close()
	ts=torch.Tensor(rs)
	ts:resize(#rs/lsize,lsize)
	file=torch.DiskFile(frs,'w')
	file:writeObject(ts)
	file:close()
end

function convfile(fsrc,frs,uselong)
	local file=io.open(fsrc)
	local lind=file:read("*n")
	local num=file:read("*n")
	local rs={}
	while num do
		local tmpt={}
		for i=1,lind do
			table.insert(tmpt,num)
			num=file:read("*n")
		end
		table.insert(rs,tmpt)
	end
	file:close()
	if uselong then
		ts=torch.LongTensor(rs)
	else
		ts=torch.Tensor(rs)
	end
	file=torch.DiskFile(frs,'w')
	file:writeObject(ts)
	file:close()
end

function gvec(nvec,vecsize,frs)
	local file=torch.DiskFile(frs,"w")
	file:writeObject(torch.randn(nvec,vecsize))
	file:close()
end

convec("wvec.txt","wvec.asc",512)

for nf=1,178558 do
	convfile("duse/rmrb"..nf.."i.txt","thd/train"..nf.."i.asc",true)
end