require 'torch'
require 'cutorch'
--require('mobdebug').start()

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
  -- Expected input as {para1,para2,...para_n} where para_i is a table of parameters, 
  -- even works for duplicated indices
  function flatten(table_of_tables)
    
    local Tensor
    local empty = true
    
    if not table_of_tables then
      return torch.Tensor()
    end
    
    for _, parameters in pairs(table_of_tables) do
      for k,value in pairs(parameters) do
        Tensor = parameters[k].new
        empty = false
        break;
      end
    end
    
    if empty then
      return torch.Tensor()
    end


    local storages = {}
    local nParameters = 0
    for _, parameters in pairs(table_of_tables) do
      for k,value in pairs(parameters) do
        local storage = parameters[k]:storage()
        if not storageInSet(storages, storage) then
          storages[torch.pointer(storage)] = {storage, nParameters}
          nParameters = nParameters + storage:size()
        end
      end
    end

    local flatParameters = Tensor(nParameters):fill(1)
    local flatStorage = flatParameters:storage()
    for _, parameters in pairs(table_of_tables) do
      for k,value in pairs(parameters) do
        local storageOffset = storageInSet(storages, parameters[k]:storage())
        parameters[k]:set(flatStorage,
          storageOffset + parameters[k]:storageOffset(),
          parameters[k]:size(),
          parameters[k]:stride())
        parameters[k]:zero()
      end
    end

    local maskParameters=  flatParameters:float():clone()
    local cumSumOfHoles = flatParameters:float():cumsum(1)
    local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
    local flatUsedParameters = Tensor(nUsedParameters)
    local flatUsedStorage = flatUsedParameters:storage()
    for _, parameters in pairs(table_of_tables) do
      for k,value in pairs(parameters) do
        local offset = cumSumOfHoles[parameters[k]:storageOffset()]
        parameters[k]:set(flatUsedStorage,
          parameters[k]:storageOffset() - offset,
          parameters[k]:size(),
          parameters[k]:stride())
      end
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


