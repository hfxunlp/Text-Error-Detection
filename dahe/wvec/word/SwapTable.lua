local SwapTable, parent = torch.class('nn.SwapTable', 'nn.Module')

function SwapTable:__init(self)
	parent:__init(self)
end

function SwapTable:updateOutput(input)
	self.output = {}
	self.input = input or self.input
	local _tmpi
	local _tmpj
	local _ninput = #self.input
	local _noutput = #self.input[1]
	for _tmpi=1,_noutput do
		table.insert(self.output,{})
	end
	for _tmpi=1,_noutput do
		for _tmpj=1,_ninput do
			table.insert(self.output[_tmpi],self.input[_tmpj][_tmpi])
		end
	end
	return self.output
end

function SwapTable:updateGradInput(input, gradOutput)
	self.gradOutput = gradOutput or self.gradOutput
	self.gradInput = {}
	local _tmpi
	local _tmpj
	local _ngradoutput = #self.gradOutput
	local _ngradinput = #self.gradOutput[1]
	for _tmpi=1,_ngradinput do
		table.insert(self.gradInput,{})
	end
	for _tmpi=1,_ngradinput do
		for _tmpj=1,_ngradoutput do
			table.insert(self.gradInput[_tmpi],self.gradOutput[_tmpj][_tmpi])
		end
	end
	return self.gradInput
end