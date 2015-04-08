
--require('mobdebug').start()

do
  local BillionWords = torch.class('BillionWords')

  
  function BillionWords:__init(opt)
    self.opt = opt
  end

  function BillionWords:load(file_path)
    local data = torch.load(file_path)
    collectgarbage()
    return data
  end
  
  --[[
    input: a raw dataset
    output: a dataset consists of 'context_size-gram'
  --]]
  function BillionWords:generateData(data, max_size)
    -- Loads data
    local input_list = {}
    local target_list = {}
    local cur_sentence = {}
    
    assert(data[data:size(1)][2] == self.opt.sentence_end_id, "data should be terminated with sentence_end_id")
    
    for i = 1, data:size(1) - 1 do
      table.insert(cur_sentence, data[i][2])
      
      local input = {}
      for j = 1, self.opt.context_size - #cur_sentence do
        table.insert(input,self.opt.sentence_start_id)
      end
      
      local target = data[i+1][2]
      
      for j = 1, #cur_sentence do
        table.insert(input, cur_sentence[j])
      end
      assert(#input == self.opt.context_size)
      table.insert(input_list,input)
      table.insert(target_list,target)
      if #input_list > max_size then
        break
      end
      if #cur_sentence == self.opt.context_size then
        table.remove(cur_sentence,1)
      end
      if data[i][2] == self.opt.sentence_end_id then
        while #cur_sentence ~= 0 do
          table.remove(cur_sentence,1)
        end
      end
    end
    local dataset=DataSet(torch.FloatTensor(input_list),torch.IntTensor(target_list),opt)
    return dataset
  end
  
  function BillionWords:loadData(data_size)
    data_size = data_size or "tiny"
    
    if data_size == "tiny" then
      self.train_data = self:load(self.opt.train_tiny)
    elseif data_size == "small" then
      self.train_data = self:load(self.opt.train_small)
    elseif data_size == "full" then
      self.train_data = self:load(self.opt.train_full)
    end
    self.valid_data = self:load(self.opt.valid_data)
    self.test_data = self:load(self.opt.test_data)
    self.word_map = self:load(self.opt.word_map)
    self.word_tree = self:load(self.opt.word_tree)
    
    -- Generates dataset for training, validation, and testing
    self.train_dataset=self:generateData(self.train_data,100000)
    self.valid_dataset=self:generateData(self.valid_data,100000)
    self.test_dataset=self:generateData(self.test_data,100000)
    return self.train_dataset,self.valid_dataset,self.test_dataset
  end
  
  function BillionWords:getSoftMaxTreeLayer()
      self.hierarchy = self:load(self.opt.word_tree)
     
      local smt = nn.SoftMaxTree(self.opt.word_embedding_size, self.hierarchy, self.opt.root_id)
      return smt
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
