--[[
DataSet Class
--]]

local class = require 'middleclass'

local DataSet = class('DataSet')

--[[
input: Tensor
target: Tensor
--]]
function DataSet:initialize(input,target,vocab,opt)
  assert(input:size(1)==target:size(1))
  self.opt=opt
  self.input=input
  self.target=target
  self.vocab=vocab
  self.data_size=input:size(1)
  self.shuffle = torch.randperm(self.data_size)
end

function getBatch(batch)
  local s=batch*(batch_size-1)+1
  local e=math.min(batch*self.opt.batch_size,self.data_size)
  local input_batch={}
  local target_batch={}
  for i=s,e do
    table.insert(input_batch,self.input[shuffle[i]])
    table.insert(target_batch,self.target[shuffle[i]])
  end
  return input_batch,target.batch
end

function shuffle()
  self.shuffle = torch.randperm(self.data_size)
end
