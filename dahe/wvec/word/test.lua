torch.setdefaulttensortype('torch.FloatTensor')
require "nn"
require "rnn"
require "SeqBLMGRU"
require "vecLookup"
require "maskZerovecLookup"
require "SwapTable"

function loadObject(fname)
	local file=torch.DiskFile(fname)
	local objRd=file:readObject()
	file:close()
	return objRd
end

function ldtt(fsrc)
	local file=io.open(fsrc)
	local lind=file:read("*n")
	local rs={}
	while lind do
		local tmpt={}
		for i=1,lind do
			num=file:read("*n")
			table.insert(tmpt,num)
		end
		table.insert(rs,tmpt)
		lind=file:read("*n")
	end
	file:close()
	return rs
end

function grs(modin,inputdata)
	local rst={}
	for i,v in ipairs(inputdata) do
		local rs=modin:forward(torch.Tensor(v):resize(#v,1))
		local tmpt={}
		for _,sequ in ipairs(rs) do
			v1,v2=unpack(sequ)
			v1:resize(v1:size(2))
			v2:resize(v2:size(2))
			table.insert(tmpt,v1:dot(v2)/(v1:norm()*v2:norm()))
		end
		table.insert(rst,tmpt)
	end
	return rst
end

function wrs(fname,tt)
	local file=io.open(fname,"w")
	for _,v in ipairs(tt) do
		for __,vwrt in ipairs(v) do
			file:write(vwrt.." ")
		end
		file:write("\n")
	end
	file:close()
end

function saveObject(fname,objWrt)
	local tmpod=nil
	if not torch.isTensor(objWrt) then
		tmpod=nn.Serial(objWrt)
		tmpod:lightSerial()
	else
		tmpod=objWrt
	end
	local file=torch.DiskFile(fname,'w')
	file:writeObject(tmpod)
	file:close()
end

print("load module")
tmod=loadObject("modrs/nnmod2.asc")
--print("Serail and save model")
--saveObject("nnmodts.asc",tmod)
tmod:evaluate()
print("load test")
td=ldtt("datasrc/tlua.txt")
print("forward")
rst=grs(tmod,td)
print("save")
wrs("modrs/trs.txt",rst)
