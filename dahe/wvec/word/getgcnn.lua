function getgcnn(ncomb,vecsize,inputdim)
	local input=nn.Identity()()
	local rgate=nn.Sigmoid()(nn.Linear(ncomb*vecsize,ncomb*vecsize)(input))
	local wcommonp=nn.CMulTable()({rgate,input})
	local wcommon=nn.Tanh()(nn.Linear(ncomb*vecsize,vecsize)(wcommonp))
	local wc=nn.JoinTable(2,2)({wcommon,input})
	local ugate=nn.SoftMax()(nn.Linear((ncomb+1)*vecsize,(ncomb+1)*vecsize)(wc))
	local www=nn.CMulTable()({ugate,wc})
	local wrs=nn.CAddTable()(nn.SplitTable(3,3)(nn.Reshape(vecsize,(ncomb+1),true)(www)))
	if inputdim>2 then
		return nn.Bottle(nn.gModule({input},{wrs}))
	else
		return nn.gModule({input},{wrs})
	end
end