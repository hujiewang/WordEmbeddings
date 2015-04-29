--[[SGD(stochastic gradient descent) class]]
do
  local SGD = torch.class('SGD')
  function SGD:__init(model,criterion,optimState)
    self._optimState = optimState
    self._model = model
    self._criterion = criterion
    self._params = {}
    self._gradParams = {}
    self._scales = {}
    local idx = 0
    for i=1,#self._model.modules do
      local param, gradParam, scale, size = self._model.modules[i]:parameters()
      local n = 0
      if param then
        for k,p in pairs(param) do
          self._params[idx+k] = p
          if gradParam then
            self._gradParams[idx+k] = gradParam[k]
          end
          if scale then
            self._scales[idx+k] = scale[k] 
          end
          n = n + 1
        end
      end
      idx = idx + n
    end
  end
  
  function SGD:zeroGradParameters()
    self._model:zeroGradParameters()
  end
  
  function SGD:updateParams()
    --[[
    -- weight decay with single or individual parameters
    if wd ~= 0 then
      dfdx:add(wd, x)
    end
    --]]
    for k, param in pairs(self._params) do
      param:add(-self._optimState.learning_rate, self._gradParams[k])  
    end
  end
  
  function SGD:optimize(inputs,targets)
    self:zeroGradParameters()
    local output = self._model:forward({inputs,targets})
    
    -- Note that TreeNLLCriterion is CPU-based, we need to copy data from GPU to CPU
    output = output:double()
    local err = self._criterion:forward(output,targets)
    
    local df_do = self._criterion:backward(output,targets)
      -- Then we need to copy data from CPU to GPU
    df_do = df_do:cuda()

    self._model:backward({inputs,targets},df_do)
    self:updateParams()
    return err
  end
end --do