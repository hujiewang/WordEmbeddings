

function predict(model,dataset,word_map,opt)
  local correct=0
  local i = 0
  local all_words=torch.IntTensor(#word_map);
  all_words:apply(function(x) i = i + 1 return i end)
  local smt_input=torch.DoubleTensor(torch.LongStorage({#word_map,opt.word_embedding_size}), torch.LongStorage({0,1}))
  
  if opt.type == 'cuda' then
    all_words = all_words:double():cuda()
    smt_input = smt_input:cuda()
  end
  
  local time = sys.clock()
  for i = 1,dataset:size() do
    -- displays progress
    xlua.progress(i,dataset:size())
    local input=dataset.input[i]
    local mlp_output = model.mlp:forward(input)
 
    smt_input[1] = mlp_output:reshape(1,opt.word_embedding_size)
    
    local err = torch.DoubleTensor(#word_map)
    for batch=1,math.ceil(#word_map/opt.predict_batch_size) do
      -- displays progress
      --xlua.progress(batch,math.ceil(#word_map/opt.predict_batch_size))
      
      local s=(batch-1)*opt.predict_batch_size+1
      local e=math.min(batch*opt.predict_batch_size,#word_map)
      
      local stm_output = model.smt:forward({smt_input[{{s,e}}], all_words[{{s,e}}]})
        -- Note that TreeNLLCriterion is CPU-based, we need to copy data from GPU to CPU
      stm_output = stm_output:double()
      err[{{s,e}}] = criterion:forward(stm_output, cur_word)
    end
    
    local predicted_word
    _,predicted_word=torch.min(err,1)
    if predicted_word[1] == dataset.target[i] then
      correct = correct + 1
    end
  end
  time = sys.clock() - time
  print("==> Speed: " .. (dataset:size()/time).. " predictions/s \n")
  return correct/dataset:size()
end