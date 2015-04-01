require 'nn'

function getModel(opt)
  
  model=nn.Sequential()
  model:add(nn.LookupTable(opt.vocab_size,opt.word_embedding_size))
  model:add(nn.Reshape(opt.context_size*opt.word_embedding_size))
  model:add(nn.Linear(opt.context_size*opt.word_embedding_size,opt.hidden_layer_size))
  model:add(nn.Tanh())
  model:add(nn.Linear(opt.hidden_layer_size,opt.output_layer_size))
  model:add(nn.LogSoftMax())
  
  criterion = nn.ClassNLLCriterion()
  
  --[[
  if opt.type == 'cuda' then
      -- Moves to CUDA
    model:cuda()
    criterion:cuda()
  end
  --]]
  return model, criterion
end