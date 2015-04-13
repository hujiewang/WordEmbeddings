

function predict(dataset,model,billionwords,opt)
  local correct=0
  local i = 0
  local all_words=torch.IntTensor(#billionwords.word_map);
  all_words:apply(function(x) i = i + 1 return i end)
  local smt_input=torch.DoubleTensor(torch.LongStorage({#billionwords.word_map,opt.word_embedding_size}), torch.LongStorage({0,1}))
  local predicted_words = torch.IntTensor(dataset:size(),5)
  local cost = torch.DoubleTensor(dataset:size(),5)
  if opt.type == 'cuda' then
    all_words = all_words:double():cuda()
    smt_input = smt_input:cuda()
  end
  
  local time = sys.clock()
  for i = 1,dataset:size() do
    -- displays progress
    print("data #"..i.." (Total "..dataset:size()..")")
    --xlua.progress(i,dataset:size())
    local input
    if dataset.input:dim() == 1 then
      input = dataset.input
    else
      input=dataset.input[i]
    end
    
    local mlp_output = model.mlp:forward(input)
 
    smt_input[1] = mlp_output:reshape(1,opt.output_layer_size)
    
    local err = torch.DoubleTensor(#billionwords.word_map)
    for batch=1,math.ceil(#billionwords.word_map/opt.predict_batch_size) do
      --displays progress
      xlua.progress(batch,math.ceil(#billionwords.word_map/opt.predict_batch_size))
      
      local s=(batch-1)*opt.predict_batch_size+1
      local e=math.min(batch*opt.predict_batch_size,#billionwords.word_map)
      
      local stm_output = model.smt:forward({smt_input[{{s,e}}], all_words[{{s,e}}]})
        -- Note that TreeNLLCriterion is CPU-based, we need to copy data from GPU to CPU
      stm_output = stm_output:double()
      err[{{s,e}}] = criterion:forward(stm_output, cur_word)
    end
    
    err, indices = torch.sort(err)
    
    -- We select top-5 predictions
    cost[i] = err[{{1,5}}]:reshape(1,5)
    predicted_words[i] = indices[{{1,5}}]:reshape(1,5)
    
    if dataset.target then
      for j=1,predicted_words[i]:size(1) do
        if predicted_words[i][j] == dataset.target[i] then
          correct = correct + 1
          break
        end
      end
    end
    
  end
  time = sys.clock() - time
  print("==> Speed: " .. (dataset:size()/time).. " predictions/s \n")
  return predicted_words,cost,correct/dataset:size()
end

--[[
  A single prediction
  Input: [table of opt.context_size strings] [model] [billionwords] [opt]
  Ouput: [string] as the predicted next word
--]]

function predictSingle(list_of_words,model,billionwords,opt)
  -- Transforms strings into an IntTensor of indices
  assert(list_of_words ~= nil)
  local input = billionwords:toIndices(list_of_words)
  local dataset=DataSet(torch.DoubleTensor(input),nil,opt)
  local predicted_words, cost, accuracy = predict(dataset,model,billionwords,opt)
  print("Predicted words:")
  for i=1,predicted_words[1]:size(1) do
    print("Predicted word: "..billionwords.word_map[predicted_words[1][i]].." Cost: "..cost[1][i].." Index: "..predicted_words[1][i])
  end
  return predicted_word, cost
end