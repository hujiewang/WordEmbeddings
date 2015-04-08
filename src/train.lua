
--require('mobdebug').start()

function train(model,criterion,dataset,opt)
  trainLogger = optim.Logger(paths.concat(opt.save,'train.log'))

  if opt.optimization == 'CG' then
    optimState = {
      maxIter = opt.maxIter
    }
    optimMethod = optim.cg

  elseif opt.optimization == 'LBFGS' then
    optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
    }
    optimMethod = optim.lbfgs

  elseif opt.optimization == 'SGD' then
    optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = 1e-7
    }
    optimMethod = optim.sgd
  else
    error('unknown optimization method')
  end
  --[[
  trainer = nn.StochasticGradient(model, criterion);
  trainer.learningRate = opt.learning_rate;
  trainer.maxIteration = 1;
  --]]
  para={}
  para.mlp={}
  para.smt={}
  para.mlp.parameters,para.mlp.gradParameters = model.mlp:parameters()
  para.smt.parameters,para.smt.gradParameters = model.smt:parameters()

  
  parameters,gradParameters = flattenParameters({para.mlp.parameters,para.smt.parameters},{para.mlp.gradParameters,para.smt.gradParameters})
 
  for epoch = 1,opt.max_epochs do
    local time = sys.clock()
    --for batch=1,opt.batch_size do
    local cost=0
    for batch=1,math.ceil(dataset:size()/opt.batch_size) do
      -- displays progress
      xlua.progress(batch,math.ceil(dataset:size()/opt.batch_size))
      
      local ds,inputs,targets = dataset:getBatch(batch)
      
      local function feval(x)
        
        --[[
        -- get new parameters
        if x~=parameters then
          parameters:copy(x)
        end
        --]]
        
        
        -- reset gradients
        gradParameters:zero()
        
        
        local mlp_output = model.mlp:forward(inputs)
        local stm_output = model.smt:forward({mlp_output,targets})
        
        -- Note that TreeNLLCriterion is CPU-based, we need to copy data from GPU to CPU
        stm_output = stm_output:double()
        local err = criterion:forward(stm_output,targets)
        
        local df_do = criterion:backward(stm_output,targets)
          -- Then we need to copy data from CPU to GPU
        df_do = df_do:cuda()
        
        local smt_grad_input = model.smt:backward({mlp_output,targets},df_do)
        
        -- SoftMaxTree backward produces {gradInput, gradTarget}, we only need gradInput
        smt_grad_input = smt_grad_input[1]
        model.mlp:backward(inputs,smt_grad_input)
        
        --gradParameters:div((#inputs)[1])
        --cost = cost/(#inputs)[1]
        cost=cost+err
        --[[
        local gotit=false
        for i=1,(#gradParameters)[1] do
          if gradParameters[i]~=0 then
            gotit=true
          end
        end
        if not gotit then
          error("no gotit")
        end
        --]]
        return err,gradParameters
      end
      optimMethod(feval,parameters,opt.optimState)

    end
    time = sys.clock() - time
    print("==> Speed: " .. (dataset:size()/time).. " samples/s \n")
    print("==> Average cost: " .. (cost/math.ceil(dataset:size()/opt.batch_size)) .. "\n")
    --trainLogger:add{["% mean class accuracy "]=confusion.totalValid*100}
    -- next epoch
    --dataset:shuffle()
  end
    local model_file=paths.concat(opt.save,"model.net")
    --os.execute('mkdir -p' .. sys.dirname(model_file))
    torch.save(model_file,model)
    print("==> Saving model completed!\n")

end

--[[
  -- test
  opt={
    -- Training hyperparameters
    type = 'int',
    optimization = 'SGD',
    learning_rate = 1e-3,
    weight_decay = 0.1,
    momentum = 0.9,
    batch_size = 1,
    loss = 'nll ',
    max_epochs=1,

    -- Data parameters
    word_embedding_size = 50,
    context_size = 5,
    vocab_size = 100,
    
    -- Model parameters
    hidden_layer_size = 10,
    output_layer_size = 100,

    -- Logger
    save="../log/"
  }

  model=nn.Sequential()
  model:add(nn.LookupTable(opt.vocab_size,opt.word_embedding_size))
  model:add(nn.Reshape(opt.context_size*opt.word_embedding_size))
  model:add(nn.Linear(opt.context_size*opt.word_embedding_size,opt.hidden_layer_size))
  model:add(nn.Tanh())
  model:add(nn.Linear(opt.hidden_layer_size,opt.output_layer_size))
  model:add(nn.LogSoftMax())

  criterion = nn.ClassNLLCriterion()
  train(model,criterion,nil,opt)
  --]]