require "nn"
require "dpnn"
require "rnn"
require "SeqBLMGRU"
require "SeqDropout"
require "vecLookup"
require "maskZerovecLookup"
require "ASequencerCriterion"

function getnn()
	local id2vec=nn.maskZerovecLookup(wvec);
	nclass=wvec:size(1)
	wvec=nil

	local coremod=nn.SeqBLMGRU(sizvec,nfeature,true);

	local clsmod=nn.Sequencer(nn.MaskZero(nn.Sequential():add(nn.Linear(nfeature,nhidden)):add(nn.Tanh()):add(nn.Linear(nhidden,nclass)):add(nn.LogSoftMax()),1));

	local nnmod=nn.Sequential():add(id2vec):add(nn.SeqDropout()):add(coremod):add(nn.SplitTable(1)):add(clsmod);

	return nnmod
end

function getar()
	return nn.SplitTable(1)
end

function getcrit()
	return nn.ASequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1));
end

function updvec(modin,svalue)
	modin:get(1).updatevec=svalue
end