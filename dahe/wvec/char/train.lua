print("set default tensor type to float")
torch.setdefaulttensortype('torch.FloatTensor')

function gradUpdate(mlpin, x, y, criterionin, learningRate)
	local pred=mlpin:forward(x)
	local err=criterionin:forward(pred, y)
	sumErr=sumErr+err
	local gradCriterion=criterionin:backward(pred, y)
	mlpin:zeroGradParameters()
	mlpin:backward(x, gradCriterion)
	mlpin:updateGradParameters(0.875)
	mlpin:updateParameters(learningRate)
	mlpin:maxParamNorm(-1)
end

function saveObject(fname,objWrt,forwls)
	if not torch.isTensor(objWrt) then
		if forwls then
			objWrt:forward(lmodi)
		end
		objWrt:lightSerial()
	end
	local file=torch.DiskFile(fname,'w')
	file:writeObject(objWrt)
	file:close()
end

print("load settings")
require"conf"

print("load data")
require "dloader"

sumErr=0
crithis={}
erate=0
storemini=1
minerrate=starterate

function train()

	print("design neural networks and criterion")
	require "designn"
	local nnmod=getnn()

	print(nnmod)
	nnmod:training()

	local critmod=getcrit()

	print("init train")
	local epochs=1
	local lr=modlr
	local eatrain=pdcycle*tld
	local culen=nil
	collectgarbage()

	print("start pre train")

	nnmod:get(1):get(1):get(1).updatevec=false

	for tmpi=1,warmcycle do
		io.write("epoch:"..tostring(epochs)..",lr:"..lr)
		for _tmpi=1,pdcycle do
			for k,v in ipairs(mword) do
				culen=v:size(1)-1
				gradUpdate(nnmod,{v:narrow(1,1,culen),v:narrow(1,2,culen)},mwordt[k],critmod,lr)
			end
		end
		local erate=sumErr/eatrain
		table.insert(crithis,erate)
		print(",Loss:"..erate)
		sumErr=0
		epochs=epochs+1
		if math.random()<prld then
			print("FTS")
			rldt()
		end
	end

	nnmod:get(1):get(1):get(1).updatevec=true

	epochs=1
	icycle=1

	aminerr=0
	lrdecayepochs=1

	while true do
		print("start innercycle:"..icycle)
		for innercycle=1,gtraincycle do
			io.write("epoch:"..tostring(epochs)..",lr:"..lr)
			for _tmpi=1,pdcycle do
				for k,v in ipairs(mword) do
					culen=v:size(1)-1
					gradUpdate(nnmod,{v:narrow(1,1,culen),v:narrow(1,2,culen)},mwordt[k],critmod,lr)
				end
			end
			local erate=sumErr/eatrain
			table.insert(crithis,erate)
			print(",Loss:"..erate)
			if erate<minerrate then
				minerrate=erate
				aminerr=0
				print("new minimal error found,save model")
				saveObject("modrs/nnmod"..storemini..".asc",nnmod,true)
				storemini=storemini+1
				if storemini>csave then
					storemini=1
				end
			else
				if aminerr>=expdecaycycle then
					aminerr=0
					if lrdecayepochs>lrdecaycycle then
						modlr=lr
						lrdecayepochs=1
					end
					lrdecayepochs=lrdecayepochs+1
					lr=modlr/(lrdecayepochs)
				end
				aminerr=aminerr+1
			end
			sumErr=0
			epochs=epochs+1
			if math.random()<prld then
				print("FTS")
				rldt()
			end
		end

		print("save neural network trained")
		saveObject("modrs/nnmod.asc",nnmod,true)

		print("save criterion history trained")
		local critensor=torch.Tensor(crithis)
		saveObject("modrs/crit.asc",critensor)

		critensor=nil

		print("task finished!Minimal error rate:"..minerrate)

		print("wait for test, neural network saved at (dev)nnmod*.asc")

		icycle=icycle+1

		print("collect garbage")
		collectgarbage()

	end
end

train()
