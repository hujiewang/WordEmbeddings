

function predict(model,dataset,word_map,opt)
  local correct=0
  
  local smt_input = torch.CudaTensor(torch.LongStorage({#word_map,opt.context_size}),torch.LongStorage({0,1})
  local all_words = torch.IntTensor(#word_map)
  all_words:apply(function(x) i = i + 1 return i end)

  for i = 1,dataset:size() do
    -- displays progress
    xlua.progress(i,dataset:size())
    local input=dataset.input[i]
    local mlp_output = model.mlp:forward(input)
    smt_input[1] = mlp_output:reshape(1,opt.word_embedding_size)
    
    local predicted_word
    local predicted_word_err = -1
    
    local stm_output = model.smt:forward({smt_input,all_words})
    -- Note that TreeNLLCriterion is CPU-based, we need to copy data from GPU to CPU
    stm_output = stm_output:double()
    local err = criterion:forward(stm_output,all_words)
    _,predicted_word=torch.max(err,1)
    if predicted_word[1] == dataset.target[i] then
      correct = correct + 1
    end
  end
  return correct/dataset:size()
end