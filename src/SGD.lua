require 'dp'
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
    self._pastGradParams = {}
    self._updateCounter = 0
    self._optimState.nesterov = self._optimState.nesterov or false
    self._optimState.learning_rate_decay = self._optimState.learning_rate_decay or 0
    self._optimState.learning_rate = self._optimState.learning_rate or 1e-3
    self._optimState.weight_decay = self._optimState.weight_decay or 0
    
    self._optimState.momentum = self._optimState.momentum or 0
    self._optimState.damp = self._optimState.damp or self._optimState.momentum
    --print(self._optimState.nesterov,self._optimState.momentum,self._optimState.damp)
    --assert(not self._optimState.nesterov or (self._optimState.momentum > 0 and self._optimState.damp == 0), "Nesterov momentum requires a momentum and zero dampening")
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
    local cur_learning_rate = self._optimState.learning_rate / (1+self._updateCounter * self._optimState.learning_rate_decay)
    print("learning_rate: "..cur_learning_rate)
    for k, gradParam in pairs(self._gradParams) do
      
      local param = self._params[k]
      -- the ordering here is important
      -- weight decay 
      if self._optimState.weight_decay~=0 then
        gradParam:add(self._optimState.weight_decay, param)
      end
      
      -- apply momentum
      if self._optimState.momentum~=0 then
        local pastGradParam = self._pastGradParams[k]
        if not pastGradParam then
          pastGradParam = torch.protoClone(gradParam, gradParam:size())
          pastGradParam:copy(gradParam)
          self._pastGradParams[k] = pastGradParam
        else
          pastGradParam:mul(self._optimState.momentum)
          pastGradParam:add(1-self._optimState.damp, gradParam)
        end
        if self._optimState.nesterov then
         gradParam:add(self._optimState.momentum, pastGradParam)
        else
         gradParam:copy(pastGradParam)
        end
      end
      
      -- learning rate decay
      
      -- gradient descent
      param:add(-cur_learning_rate, gradParam)  
    end
    self._updateCounter = self._updateCounter + 1
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