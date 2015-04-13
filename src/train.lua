
--require('mobdebug').start()

function train(model,criterion,dataset,opt)
  trainLogger = optim.Logger(paths.concat(opt.save,'train.log'))
  
  -- For early stopping
  local last_valid_accuracy = 0.0
  
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
        
        cost=cost+err
        return err,gradParameters
      end
      optimMethod(feval,parameters,opt.optimState)

    end
    time = sys.clock() - time
    print("==> Speed: " .. (dataset:size()/time).. " samples/s \n")
    print("==> Average cost: " .. (cost/math.ceil(dataset:size()/opt.batch_size)) .. "\n")
    
    if epoch % opt.valid_time_gap ==0 then
      _,_,accuracy = predict(valid_dataset,model,billionwords,opt)
      trainLogger:add{["% top-5 class accuracy "]=accuracy}
      if accuracy < last_valid_accuracy and math.abs(accuracy - last_valid_accuracy) >= opt.earlyStopping_threshold then
        print("==> Early Stopper terminated training")
        break
      end
    end
    -- next epoch
    --dataset:shuffle()
  end
    local model_file=paths.concat(opt.save,"model.net")
    --os.execute('mkdir -p' .. sys.dirname(model_file))
    torch.save(model_file,model)
    print("==> Saving model completed!\n")

end
