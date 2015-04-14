-- A neat script that is useful for testing and visualizations
require 'main'



--require('mobdebug').start()

--[[
  Input: a string
  Output: table of 6 nearest neighbours {{string,distance}..}
--]]
function neighbors(word)
  word_index = billionwords.index_map[word]
  
  if word_index == nil then
    print("["..word.."] does not exist in the dictionary!")
    return nil
  end
  
  distance = torch.DoubleTensor(#billionwords.word_map)
  weight = model.mlp.modules[1].weight[word_index]
  for i = 1,#billionwords.word_map do
    if i%100000 ==0 then
      print(i)
    end
    --xlua.progress(i,#billionwords.word_map)
    tmp = weight - model.mlp.modules[1].weight[i]
    distance[i] = torch.sum(tmp:cmul(tmp))
  end
  distance:sqrt()
  dist, indices = torch.sort(distance)
  
  result={}
  for i=1,6 do
    table.insert(result,{billionwords.word_map[indices[i]],dist[i]})
  end
  return result
end

function test()
  print("==> Loading model...")
  
  model = torch.load("../log/model.net")
  billionwords = BillionWords(opt)
  billionwords:loadWordMap()
  
  while true do
    print("==> Ready for input")
    local word = io.read()
    print("==> Calculating neighbors for ["..word.."]")
    rv = neighbors(word)
    for k,v in pairs(rv) do
      print(v[1].." dist: "..v[2])
    end
  end
end

test()