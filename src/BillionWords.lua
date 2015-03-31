require 'torch'
require 'xlua'

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

  function BillionWords:loadData(data_size)
    data_size = data_size or "tiny"
    
    if data_size == "tiny" then
      self.train = self:load(self.opt.train_tiny)
    elseif data_size == "small" then
      self.train = self:load(self.opt.train_small)
    elseif data_size == "full" then
      self.train = self:load(self.opt.train_full)
    end
    self.valid = self:load(self.opt.valid_data)
    self.test = self:load(self.opt.test_data)
    self.word_map = self:load(self.opt.word_map)
    self.word_tree = self:load(self.opt.word_tree)
    
    for i = 1,150 do
      io.write(self.word_map[self.train[i][2]].." ")
      if self.word_map[self.train[i][2]] == "</S>" then
        io.write("\n")
      end
    end
  end
end
-- Tests
billionwords_opt = {
  word_map = "../data/billionwords/word_map.th7",
  test_data = "../data/billionwords/test_data.th7",
  valid_data = "../data/billionwords/valid_data.th7",
  train_tiny = "../data/billionwords/train_tiny.th7",
  train_small = "../data/billionwords/train_small.th7",
  train_full = "../data/billionwords/train_full.th7",
  word_tree = "../data/billionwords/word_tree1.th7"
}
test = BillionWords(billionwords_opt)
test:loadData()
