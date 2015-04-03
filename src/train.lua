
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

  for epoch = 1,opt.max_epochs do
    local time = sys.clock()
    --for batch=1,opt.batch_size do
    for batch=1,math.ceil(dataset:size()/opt.batch_size) do
      -- displays progress
      xlua.progress(batch,math.ceil(dataset:size()/opt.batch_size))
      collectgarbage()
      
      local inputs,targets = dataset:getBatch(batch)

      local function feval(x)
        -- get new parameters
        if x~=parameters then
          parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        local cost=0

        for i=1,(#inputs)[1] do     
          --targets[i]=3
          --print(inputs[i])
          --print(targets[i])
          --error()
          local output = model:forward(inputs[i])
          --print(output)
          --print(targets[i])
          --local err = criterion:forward(output,targets[i])
          local err = criterion:forward(output,5)
          cost = cost + err
          --print(err)
          --local df_do=criterion:backward(output,targets[i])
          local df_do=criterion:backward(output,5)
          model:backward(inputs[i],df_do)

          --confusion:add(output,targets[i])
        end

        gradParameters:div((#inputs)[1])
        cost = cost/(#inputs)[1]
        return cost,gradParameters
      end
      
      parameters,gradParameters = model:getParameters();
      --optimMethod(feval,parameters,opt.optimState)
      optim.sgd(feval,parameters,opt.optimState)
    end
    time = sys.clock() - time
    time = time / dataset:size()
    print("==> time to learn 1 sample = " .. (time*1000) .. "ms\n")
    --print("==> mean class accuracy = " .. (confusion.totalValid*100) .."%\n")
    --print(confusion)
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