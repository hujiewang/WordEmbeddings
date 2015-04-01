--[[
DataSet Class
--]]

do
  local DataSet = torch.class('DataSet')

--[[
input: Tensor
target: Tensor
--]]
  function DataSet:__init(input,target,opt)
    assert(input:size(1)==target:size(1))
    self.input=input
    self.target=target
    self.data_size=input:size(1)
    self.shuffle = torch.randperm(self.data_size)
    self.opt=opt
    
    if self.opt.type == 'double' then 
      self.input = self.input:double()
    elseif self.opt.type == 'int' then 
      self.input = self.input:int() 
    elseif self.opt.type == 'cuda' then 
      self.input = self.input:cuda() 
    end
    
  end

  function DataSet:getBatch(batch)
    local s=batch*(self.opt.batch_size-1)+1
    local e=math.min(batch*self.opt.batch_size,self.data_size)
    local input_batch={}
    local target_batch={}
    for i=s,e do
      table.insert(input_batch,self.input[self.shuffle[i]])
      table.insert(target_batch,self.target[self.shuffle[i]])
    end
    return input_batch,target_batch
  end

  function DataSet:shuffle()
    self.shuffle = torch.randperm(self.data_size)
  end
  
  function DataSet:size()
    return (#self.input)[1]
  end
end

