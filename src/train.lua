require 'torch'
require 'optim'
require 'xlua'


function train(model,criterion,dataset,opt)
  trainLogger = optim.Logger(paths.concat(opt.save,'train.log'))
  classes={}
  for i=1,opt.vocab.size do
    table.insert(classes,i)
  end
  confusion = optim.ConfusionMatrix(classes)

  if model then
    parameters,gradParameters=model:getParameters()
  end

  if opt.type == 'cuda' then
    model:cuda()
    criterion:cuda()
  end

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

    for epoch = 1,opt.max_epochs do
      local time = sys.clock()
      for batch=1,opt.batch_size do

        -- displays progress
        xlua.progress(batch,opt.batch_size)

        local input,target = dataset:getBatch(batch)

        local function feval(x)
          -- get new parameters
          if x~=parameters then
            parameters:copy(x)
          end

          -- reset gradients
          gradParameters:zero()

          local cost=0

          for i=1,#input do
            local output = model:forward(input[i])
            local err = criterion:forward(output,target[i])
            cost = cost + err

            local df_do=criterion:backward(output,target[i])
              model:backward(inputs[i],df_do)

              confusion:add(output,targets[i])
          end

          gradParameters:div(#input)
          cost = cost/#inputs
          return cost,gradParameters
        end

          optimMethod(feval,parameters,opt.optimState)
        end
        time = sys.clock() - time
        time = time / opt.data_size
        print("==> time to learn 1 sample = " .. (time*1000) .. "ms\n")
        print(confusion)
        train_logger:add{["% mean class accuracy "]=confusion.totalValid*100}
        -- next epoch
        confusion:zero()
        dataset:shuffle()
      end
      local model_file=paths.concat(opt.save,"model.net")
      os.execute('mkdir -p' .. sys.dirname(model_file))
      torch.save(model_file,model)
      print("==> Saving model completed!\n")

    end