function saveObject(fname,objWrt,lmodi,bwdc)
	if not torch.isTensor(objWrt) then
		if lmodi then
			objWrt:forward(lmodi)
			objWrt:updateGradInput(lmodi,bwdc)
		end
		objWrt:lightSerial()
	end
	local file=torch.DiskFile(fname,'w')
	file:writeObject(objWrt)
	file:close()
end