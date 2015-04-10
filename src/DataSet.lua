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
    if target then
      assert(input:size(1)==target:size(1))
    end
    self.input=input
    self.target=target
    if self.input:dim() == 1 then
      self.input = self.input:reshape(1,self.input:size(1))
    end
    
    -- Reduce data size for testing
    self.input=self.input[{{1,math.min(1000,(#self.input)[1])}}]
    if self.target then 
      self.target=self.target[{{1,math.min(1000,(#self.input)[1])}}]
    end
    
    self.shuffle = torch.randperm((#self.input)[1])
    self.opt=opt
    --print("DataSet: Switching to "..self.opt.type.."\n")
    if self.opt.type == 'cuda' then
        self.input = self.input:cuda()
        if self.target then
          self.target = self.target:double():cuda()
        end
    elseif self.opt.type == 'float' then
        self.input = self.input:float()
    elseif self.opt.type == 'double' then
        self.input = self.input:double()
    end
    
  end

  function DataSet:getBatch(batch)
    local s=(batch-1)*self.opt.batch_size+1
    local e=math.min(batch*self.opt.batch_size,self:size())
    local inputs=self.input[{{s,e}}]
    local targets=self.target[{{s,e}}]
    dataset = {};
    function dataset:size() return e-s+1 end 
    for i=1,dataset:size() do
      local input = inputs[i];  
      local output = targets[i];
      dataset[i] = {input, output}
    end
    return dataset,inputs,targets
  end

  function DataSet:shuffle()
    self.shuffle = torch.randperm(self.data_size)
  end
  
  function DataSet:size()
    return (#self.input)[1]
  end
end

