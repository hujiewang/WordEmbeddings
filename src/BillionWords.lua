
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
    local dataset=DataSet(torch.DoubleTensor(input_list),torch.IntTensor(target_list),opt)
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
    self.index_map={}
    for key,value in pairs(self.word_map) do
      self.index_map[value]=key
    end
    -- Generates dataset for training, validation, and testing
    self.train_dataset=self:generateData(self.train_data,self.opt.max_train_size)
    self.valid_dataset=self:generateData(self.valid_data,self.opt.max_valid_size)
    self.test_dataset=self:generateData(self.test_data,self.opt.max_test_size)
    return self.train_dataset,self.valid_dataset,self.test_dataset
  end
  
  function BillionWords:getSoftMaxTreeLayer()
      self.hierarchy = self:load(self.opt.word_tree)
     
      local smt = nn.SoftMaxTree(self.opt.word_embedding_size, self.hierarchy, self.opt.root_id)
      return smt
  end
  
  function BillionWords:toIndices(list_of_words)
    --assert(list_of_words ~= nil)
    indices = {}
    for i = 1, #list_of_words do
      indices[i]=self.index_map[list_of_words[i]]
    end
    return indices
  end
end
