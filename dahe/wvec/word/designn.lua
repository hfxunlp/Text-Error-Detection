function getnn()
	local id2vec=nn.maskZerovecLookup(wvec);

	local coremod=nn.SeqBGRU(sizvec,nfeature);
	coremod.maskzero=true
	--coremod.batchfirst=true	

	local clsmod=nn.Sequencer(nn.MaskZero(nn.Sequential():add(nn.Linear(nfeature,nhidden)):add(nn.Tanh()):add(nn.Linear(nhidden,nclass)),1));

	local nnmodi=nn.Sequential():add(id2vec):add(coremod):add(nn.SplitTable(1)):add(clsmod);
	local nnmod=nn.ParallelTable():add(nnmodi):add(nn.Identity())

	return nnmod
end

function getcrit()
	return nn.SequencerCriterion(nn.MaskZeroCriterion(nn.CosineEmbeddingCriterion(),1));
end
