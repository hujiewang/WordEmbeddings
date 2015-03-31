--[[
DataSet Class
--]]

do
  local DataSet = torch.class('DataSet')

--[[
input: Tensor
target: Tensor
--]]
  function DataSet:__init(input,target)
    assert(input:size(1)==target:size(1))
    self.input=input
    self.target=target
    self.data_size=input:size(1)
    self.shuffle = torch.randperm(self.data_size)
  end

  function getBatch(batch, batch_size)
    local s=batch*(batch_size-1)+1
    local e=math.min(batch*self.opt.batch_size,self.data_size)
    local input_batch={}
    local target_batch={}
    for i=s,e do
      table.insert(input_batch,self.input[shuffle[i]])
      table.insert(target_batch,self.target[shuffle[i]])
    end
    return input_batch,target_batch
  end

  function shuffle()
    self.shuffle = torch.randperm(self.data_size)
  end
end
