require "nn"
require "dpnn"
require "rnn"
require "SeqBLMGRU"
require "SeqDropout"
require "vecLookup"
require "maskZerovecLookup"
require "getgcnn"
require "ASequencerCriterion"

function getnn()
	local id2vec=nn.maskZerovecLookup(wvec);
	wvec=nil

	local coremod=nn.SeqBLMGRU(sizvec,sizvec*2,true);

	local clsmod=nn.Sequencer(nn.MaskZero(getgcnn(2,sizvec,2),1));

	local nnmodi=nn.Sequential():add(coremod):add(nn.SplitTable(1)):add(clsmod);
	local nnmod=nn.Sequential():add(id2vec):add(nn.SeqDropout()):add(nn.ConcatTable():add(nnmodi):add(nn.SplitTable(1))):add(nn.ZipTable())

	return nnmod
end

function getcrit()
	return nn.ASequencerCriterion(nn.MaskZeroCriterion(nn.CosineEmbeddingCriterion(),1));
end

function updvec(modin,svalue)
	modin:get(1).updatevec=svalue
end