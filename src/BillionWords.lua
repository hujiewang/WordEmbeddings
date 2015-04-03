
--require('mobdebug').start()

do
  local BillionWords = torch.class('BillionWords')
  
  --[[
  b_opt={
    train_tiny
    train_small
    train_full
    valid_data
    test_data
    word_map
    context_size
    sentence_start_id
    sentence_end_id
  }
  --]]
  
  function BillionWords:__init(b_opt, opt)
    self.opt = opt
    self.b_opt = b_opt
  end

  function BillionWords:load(file_path)
    local data = torch.load(file_path)
    collectgarbage()
    return data
  end

  function BillionWords:loadData(data_size)
    data_size = data_size or "tiny"
    
    if data_size == "tiny" then
      self.train_data = self:load(self.b_opt.train_tiny)
    elseif data_size == "small" then
      self.train_data = self:load(self.b_opt.train_small)
    elseif data_size == "full" then
      self.train_data = self:load(self.b_opt.train_full)
    end
    self.valid = self:load(self.b_opt.valid_data)
    self.test = self:load(self.b_opt.test_data)
    self.word_map = self:load(self.b_opt.word_map)
    self.word_tree = self:load(self.b_opt.word_tree)
    
    -- Loads data
    local input_list = {}
    local target_list = {}
    local cur_sentence = {}
    
    assert(self.train_data[self.train_data:size(1)][2] == self.b_opt.sentence_end_id, "train_data should be terminated with sentence_end_id")
    
    for i = 1, self.train_data:size(1) - 1 do
      table.insert(cur_sentence, self.train_data[i][2])
      
      local input = {}
      for j = 1, self.b_opt.context_size - #cur_sentence do
        table.insert(input,self.b_opt.sentence_start_id)
      end
      
      local target = self.train_data[i+1][2]
      
      for j = 1, #cur_sentence do
        table.insert(input, cur_sentence[j])
      end
      assert(#input == self.b_opt.context_size)
      table.insert(input_list,input)
      table.insert(target_list,target)
      
      if #cur_sentence == self.b_opt.context_size then
        table.remove(cur_sentence,1)
      end
      if self.train_data[i][2] == self.b_opt.sentence_end_id then
        while #cur_sentence ~= 0 do
          table.remove(cur_sentence,1)
        end
      end
    end
    local dataset=DataSet(torch.Tensor(input_list),torch.Tensor(target_list),opt)
    return dataset
  end
  
  
end


--[[
--Tests

billionwords_opt = {
  word_map = "../data/billionwords/word_map.th7",
  test_data = "../data/billionwords/test_data.th7",
  valid_data = "../data/billionwords/valid_data.th7",
  train_tiny = "../data/billionwords/train_tiny.th7",
  train_small = "../data/billionwords/train_small.th7",
  train_full = "../data/billionwords/train_full.th7",
  word_tree = "../data/billionwords/word_tree1.th7",
  context_size = 5,
  sentence_start_id = 793470,
  sentence_end_id = 793471,
  sentence_unknown_id = 793469,
  root_id = 880542
}
test = BillionWords(billionwords_opt, opt)
dataset = test:loadData("tiny")
print(#test.word_map)
--]]
