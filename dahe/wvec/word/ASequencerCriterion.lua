------------------------------------------------------------------------
--[[ ASequencerCriterion ]]--
-- Applies a criterion to each of the inputs and targets in the 
-- corresponding input and target Tables.
-- Useful for nn.Repeater and nn.Sequencer.
-- WARNING : assumes that the decorated criterion is stateless, i.e. 
-- the backward doesn't need to be preceded by a commensurate forward.
------------------------------------------------------------------------
local ASequencerCriterion, parent = torch.class('nn.ASequencerCriterion', 'nn.Criterion')

function ASequencerCriterion:__init(criterion)
   parent.__init(self)
   self.criterion = criterion
   if torch.isTypeOf(criterion, 'nn.ModuleCriterion') then
      error("ASequencerCriterion shouldn't decorate a ModuleCriterion. "..
         "Instead, try the other way around : "..
         "ModuleCriterion decorates a ASequencerCriterion. "..
         "Its modules can also be similarly decorated with a Sequencer.")
   end
   self.clones = {}
   self.gradInput = {}
end

function ASequencerCriterion:getStepCriterion(step)
   assert(step, "expecting step at arg 1")
   local criterion = self.clones[step]
   if not criterion then
      criterion = self.criterion:clone()
      self.clones[step] = criterion
   end
   return criterion
end

function ASequencerCriterion:updateOutput(input, target)
   self.output = 0
   local nStep
   if torch.isTensor(input) then
      assert(torch.isTensor(target), "expecting target Tensor since input is a Tensor")
      assert(target:size(1) == input:size(1), "target should have as many elements as input")
      nStep = input:size(1)
   else
      assert(torch.type(target) == 'table', "expecting target table")
      assert(#target == #input, "target should have as many elements as input")
      nStep = #input
   end

   
   for i=1,nStep do
      local criterion = self:getStepCriterion(i)
      self.output = self.output + criterion:forward(input[i], target[i])
   end
   
   self.output = self.output/nStep
   
   return self.output
end

function ASequencerCriterion:updateGradInput(input, target)
   self.gradInput = {}
   if torch.isTensor(input) then
      assert(torch.isTensor(target), "expecting target Tensor since input is a Tensor")
      assert(target:size(1) == input:size(1), "target should have as many elements as input")
      nStep = input:size(1)
   else
      assert(torch.type(target) == 'table', "expecting gradOutput table")
      assert(#target == #input, "target should have as many elements as input")
      nStep = #input
   end
   
   local tableGradInput = {}
   for i=1,nStep do
      local criterion = self:getStepCriterion(i)
      tableGradInput[i] = criterion:backward(input[i], target[i])
   end
   
   if torch.isTensor(input) then
      self.gradInput = tableGradInput[1].new()
      self.gradInput:resize(nStep, unpack(tableGradInput[1]:size():totable()))
      for step=1,nStep do
         self.gradInput[step]:copy(tableGradInput[step])
      end
   else
      self.gradInput = tableGradInput
   end
   
   return self.gradInput
end