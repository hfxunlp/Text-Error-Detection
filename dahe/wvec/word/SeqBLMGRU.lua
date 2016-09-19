------------------------------------------------------------------------
--[[ SeqBLMGRU ]] --
-- Bi-directional RNN using two SeqGRU modules.
-- Input is a tensor e.g time x batch x inputdim.
-- Output is a tensor of the same length e.g time x batch x outputdim.
-- Applies a forward rnn to input tensor in forward order
-- and applies a backward rnn in reverse order.
-- Reversal of the sequence happens on the time dimension.
------------------------------------------------------------------------
local SeqBLMGRU, parent = torch.class('nn.SeqBLMGRU', 'nn.Container')

function SeqBLMGRU:__init(inputDim, hiddenDim, maskzero)

	parent.__init(self)

	self.forwardModule = nn.SeqGRU(inputDim, hiddenDim)

	if maskzero then
		self.forwardModule.maskzero=true
	end

	self.backwardModule = nn.Sequential()
		:add(nn.SeqReverseSequence(1)) -- reverse
		:add(self.forwardModule:clone())
		:add(nn.SeqReverseSequence(1)) -- unreverse

	self.modules={self.forwardModule,self.backwardModule}

end

function SeqBLMGRU:updateOutput(input)
	self.input = input or self.input
	local _pseql = self.input:size(1)-1
	local _tmptf = self.forwardModule:updateOutput(self.input:narrow(1,1,_pseql))
	self.output = torch.zeros(_tmptf:narrow(1,1,1):size()):cat(_tmptf,1)
	self.output:narrow(1,1,_pseql):add(self.backwardModule:updateOutput(self.input:narrow(1,2,_pseql)))
	return self.output
end

function SeqBLMGRU:updateGradInput(input, gradOutput)
	self.input = input or self.input
	self.gradOutput = gradOutput or self.gradOutput
	local _pseql = self.gradOutput:size(1)-1
	local _tmptf = self.forwardModule:updateGradInput(self.input:narrow(1,1,_pseql),self.gradOutput:narrow(1,2,_pseql))
	self.gradInput = torch.zeros(_tmptf:narrow(1,1,1):size()):cat(_tmptf,1)
	self.gradInput:narrow(1,1,_pseql):add(self.backwardModule:updateGradInput(self.input:narrow(1,2,_pseql),self.gradOutput:narrow(1,1,_pseql)))
	return self.gradInput
end

function SeqBLMGRU:accGradParameters(input, gradOutput, scale)
	self.input = input or self.input
	self.gradOutput = gradOutput or self.gradOutput
	local _pseql = self.gradOutput:size(1)-1
	self.forwardModule:accGradParameters(self.input:narrow(1,1,_pseql), gradOutput:narrow(1,2,_pseql), scale)
	self.backwardModule:accGradParameters(self.input:narrow(1,2,_pseql), gradOutput:narrow(1,1,_pseql), scale)
end

function SeqBLMGRU:accUpdateGradParameters(input, gradOutput, lr)
	self.input = input or self.input
	self.gradOutput = gradOutput or self.gradOutput
	local _pseql = self.gradOutput:size(1)-1
	self.forwardModule:accUpdateGradParameters(self.input:narrow(1,1,_pseql), gradOutput:narrow(1,2,_pseql), lr)
	self.backwardModule:accUpdateGradParameters(self.input:narrow(1,2,_pseql), gradOutput:narrow(1,1,_pseql), lr)
end

function SeqBLMGRU:sharedAccUpdateGradParameters(input, gradOutput, lr)
	self.input = input or self.input
	self.gradOutput = gradOutput or self.gradOutput
	local _pseql = self.gradOutput:size(1)-1
	self.forwardModule:sharedAccUpdateGradParameters(self.input:narrow(1,1,_pseql), gradOutput:narrow(1,2,_pseql), lr)
	self.backwardModule:sharedAccUpdateGradParameters(self.input:narrow(1,2,_pseql), gradOutput:narrow(1,1,_pseql), lr)
end

function SeqBLMGRU:type(type, ...)
	return parent.type(self, type, ...)
end