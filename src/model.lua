
function makeParallel(layer)
  pt = nn.ParallelTable()
  pt:add(layer)
  pt:add(nn.Identity())
  return pt
end

function getModel(opt,billionwords)
  
  model = {}
  model = nn.Sequential()
  model:add(makeParallel(nn.LookupTable(opt.vocab_size,opt.word_embedding_size)))
  model:add(makeParallel(nn.Reshape(opt.context_size*opt.word_embedding_size)))
  model:add(makeParallel(nn.Dropout()))
  model:add(makeParallel(nn.Linear(opt.context_size*opt.word_embedding_size,opt.hidden_layer_size)))
  model:add(makeParallel(nn.Tanh()))
  model:add(makeParallel(nn.Dropout()))
  model:add(makeParallel(nn.Linear(opt.hidden_layer_size,opt.output_layer_size)))
  model:add(makeParallel(nn.Tanh()))
  model:add(billionwords:getSoftMaxTreeLayer())

  criterion = nn.TreeNLLCriterion()
  
  if opt.type == 'cuda' then
      -- Moves to CUDA
    model:cuda()
  end
  
  return model, criterion
end