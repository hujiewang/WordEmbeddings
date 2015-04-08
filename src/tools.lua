require 'torch'
require 'cutorch'
require('mobdebug').start()

function deepcopy(orig)
  local orig_type = type(orig)
  local copy
  if orig_type == 'table' then
    copy = {}
    for orig_key, orig_value in next, orig, nil do
      copy[deepcopy(orig_key)] = deepcopy(orig_value)
    end
    setmetatable(copy, deepcopy(getmetatable(orig)))
  else -- number, string, boolean, etc
    copy = orig
  end
  return copy
end

--[[
  Flattens parameters (works even for non-contiguous tables)
--]]
function flattenParameters(parameters,gradParameters)
  --[[
  for k,value in pairs(parameters) do
    print(parameters[k]:size())
  end
  --]]
  local function storageInSet(set, storage)
    local storageAndOffset = set[torch.pointer(storage)]
    if storageAndOffset == nil then
      return nil
    end
    local _, offset = unpack(storageAndOffset)
    return offset
  end

  -- this function flattens arbitrary lists of parameters,
  -- even complex shared ones
  function flatten(parameters)
    
    local Tensor
    local empty = true
    
    if not parameters then
      return torch.Tensor()
    end
    
    for k,value in pairs(parameters) do
      Tensor = parameters[k].new
      empty = false
      break;
    end
    
    if empty then
      return torch.Tensor()
    end


    local storages = {}
    local nParameters = 0
    for k,value in pairs(parameters) do
      local storage = parameters[k]:storage()
      if not storageInSet(storages, storage) then
        storages[torch.pointer(storage)] = {storage, nParameters}
        nParameters = nParameters + storage:size()
      end
    end

    local flatParameters = Tensor(nParameters):fill(1)
    local flatStorage = flatParameters:storage()

    for k,value in pairs(parameters) do
      local storageOffset = storageInSet(storages, parameters[k]:storage())
      parameters[k]:set(flatStorage,
        storageOffset + parameters[k]:storageOffset(),
        parameters[k]:size(),
        parameters[k]:stride())
      parameters[k]:zero()
    end

    local maskParameters=  flatParameters:float():clone()
    local cumSumOfHoles = flatParameters:float():cumsum(1)
    local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
    local flatUsedParameters = Tensor(nUsedParameters)
    local flatUsedStorage = flatUsedParameters:storage()

    for k,value in pairs(parameters) do
      local offset = cumSumOfHoles[parameters[k]:storageOffset()]
      parameters[k]:set(flatUsedStorage,
        parameters[k]:storageOffset() - offset,
        parameters[k]:size(),
        parameters[k]:stride())
    end

    for _, storageAndOffset in pairs(storages) do
      local k, v = unpack(storageAndOffset)
      flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
    end

    if cumSumOfHoles:sum() == 0 then
      flatUsedParameters:copy(flatParameters)
    else
      local counter = 0
      for k = 1,flatParameters:nElement() do
        if maskParameters[k] == 0 then
          counter = counter + 1
          flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
        end
      end
      assert (counter == nUsedParameters)
    end
    return flatUsedParameters
  end

  -- flatten parameters and gradients
  local flatParameters = flatten(parameters)
  collectgarbage()
  local flatGradParameters = flatten(gradParameters)
  collectgarbage()

  -- return new flat vector that contains all discrete parameters
  return flatParameters, flatGradParameters
end


-- Tests

a={}
a[2]=torch.CudaTensor(3):fill(0)
a[10]=torch.CudaTensor(2):fill(0)
a[29]=torch.CudaTensor(1):fill(0)
b={}
b[2]=torch.CudaTensor(3):fill(0)
b[10]=torch.CudaTensor(2):fill(0)
b[29]=torch.CudaTensor(1):fill(0)

aa,bb=flattenParameters(a,b)

c={}
c[2]=torch.CudaTensor(3):fill(0)
c[10]=torch.CudaTensor(2):fill(0)
c[29]=torch.CudaTensor(1):fill(0)
d={}
d[2]=torch.CudaTensor(3):fill(0)
d[10]=torch.CudaTensor(2):fill(0)
d[29]=torch.CudaTensor(1):fill(0)

cc,dd=flattenParameters(c,d)

a[2][1]=2000
d[2][1]=2001

ee,ff=flattenParameters({aa,cc},{bb,dd})

a[2][1]=1999
d[2][1]=1998
aa[3]=-3
dd[4]=-4
ccc=3
