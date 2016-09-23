print("set default tensor type to float")
torch.setdefaulttensortype('torch.FloatTensor')

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
	require "trapi"
	require "sapi"

	print("design neural networks and criterion")
	require "designn"
	local nnmod=getnn()
	
	local tarnnmod=getar()

	print(nnmod)
	nnmod:training()

	local critmod=getcrit()

	print("init train")
	local epochs=1
	local lr=modlr
	local eatrain=pdcycle*tld
	collectgarbage()

	print("start pre train")

	updvec(nnmod,false)

	for tmpi=1,warmcycle do
		io.write("epoch:"..tostring(epochs)..",lr:"..lr)
		for _tmpi=1,pdcycle do
			for _,v in ipairs(mword) do
				gradUpdate(nnmod,v,tarnnmod:forward(v),critmod,lr)
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

	updvec(nnmod,true)

	epochs=1
	icycle=1

	aminerr=0
	lrdecayepochs=1

	while true do
		print("start innercycle:"..icycle)
		for innercycle=1,gtraincycle do
			io.write("epoch:"..tostring(epochs)..",lr:"..lr)
			for _tmpi=1,pdcycle do
				for _,v in ipairs(mword) do
					gradUpdate(nnmod,v,tarnnmod:forward(v),critmod,lr)
				end
			end
			local erate=sumErr/eatrain
			table.insert(crithis,erate)
			print(",Loss:"..erate)
			if erate<minerrate then
				minerrate=erate
				aminerr=0
				print("new minimal error found,save model")
				saveObject("modrs/nnmod"..storemini..".asc",nnmod)
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
		saveObject("modrs/nnmod.asc",nnmod)

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
