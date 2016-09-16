require "nn"
require "rnn"
require "SeqBGRU"
require "dpnn"
require "vecLookup"
require "maskZerovecLookup"
require "SwapTable"

function getnn()
	local id2vec=nn.maskZerovecLookup(wvec);

	local coremod=nn.SeqBGRU(sizvec,nfeature);
	coremod.maskzero=true
	--coremod.batchfirst=true	

	local clsmod=nn.Sequencer(nn.MaskZero(nn.Sequential():add(nn.Linear(nfeature,nhidden)):add(nn.Tanh()):add(nn.Linear(nhidden,nclass)),1));

	local nnmodi=nn.Sequential():add(id2vec):add(coremod):add(nn.SplitTable(1)):add(clsmod);
	local nnmod=nn.Sequential():add(nn.ParallelTable():add(nnmodi):add(nn.Sequential():add(id2vec):add(nn.SplitTable(1)))):add(nn.SwapTable())

	return nnmod
end

function getcrit()
	return nn.SequencerCriterion(nn.MaskZeroCriterion(nn.CosineEmbeddingCriterion(),1));
end
