function saveObject(fname,objWrt)
	local tmpod=nil
	if not torch.isTensor(objWrt) then
		tmpod=nn.Serial(objWrt)
		tmpod:mediumSerial()
	else
		tmpod=objWrt
	end
	local file=torch.DiskFile(fname,'w')
	file:writeObject(tmpod)
	file:close()
end