

function getModel(opt,billionwords)
  
  model = {}
  model.mlp = nn.Sequential()
  model.mlp:add(nn.LookupTable(opt.vocab_size,opt.word_embedding_size))
  model.mlp:add(nn.Reshape(opt.context_size*opt.word_embedding_size))
  model.mlp:add(nn.Dropout())
  model.mlp:add(nn.Linear(opt.context_size*opt.word_embedding_size,opt.hidden_layer_size))
  model.mlp:add(nn.Tanh())
  model.mlp:add(nn.Dropout())
  model.mlp:add(nn.Linear(opt.hidden_layer_size,opt.output_layer_size))
  model.mlp:add(nn.Tanh())
  model.smt = billionwords:getSoftMaxTreeLayer()

  criterion = nn.TreeNLLCriterion()
  
  if opt.type == 'cuda' then
      -- Moves to CUDA
    model.mlp:cuda()
    model.smt:cuda()
  end
  
  return model, criterion
end