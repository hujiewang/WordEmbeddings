
--require('mobdebug').start()

function train(model,criterion,dataset,opt)
  trainLogger = optim.Logger(paths.concat(opt.save,'train.log'))
  
  -- For early stopping
  local last_valid_accuracy = 0.0
  local accuracy
  local last_avg_cost = math.huge
  local avg_cost
  
  if opt.optimization == 'SGD' then
    optimState = {
      learning_rate = opt.learning_rate,
      weight_decay= opt.weight_decay,
      momentum = opt.momentum,
      learning_rate_decay = 1e-5
    }
    sgd = SGD(model,criterion,optimState)
  else
    error('unknown optimization method')
  end
 
  for epoch = 1,opt.max_epochs do
    local time = sys.clock()
    --for batch=1,opt.batch_size do
    local cost=0
    for batch=1,math.ceil(dataset:size()/opt.batch_size) do
      -- displays progress
      xlua.progress(batch,math.ceil(dataset:size()/opt.batch_size))
      
      local ds,inputs,targets = dataset:getBatch(batch)
      cost = cost + sgd:optimize(inputs,targets)
    end
    time = sys.clock() - time
    avg_cost = (cost/math.ceil(dataset:size()/opt.batch_size))
    print("==> Speed: " .. (dataset:size()/time).. " samples/s")
    print("==> Average cost: " ..avg_cost)
    print("==> Average cost change: "..(avg_cost-last_avg_cost))
    print("==> Epoch #"..epoch.." completed (Max number of epochs "..opt.max_epochs..")")
    
    if epoch % opt.valid_time_gap ==0 then
      print("==> Validating...")
      _,_,accuracy = predict(valid_dataset,model,billionwords,opt)
      print("==> Validation top-5 class accuracy: " ..accuracy)
      trainLogger:add{["% top-5 class accuracy "]=accuracy}
      if accuracy < last_valid_accuracy and math.abs(accuracy - last_valid_accuracy) >= opt.earlyStopping_threshold then
        print("==> Early Stopper terminated training")
        break
      end
      last_valid_accuracy = accuracy
    end

    last_avg_cost = avg_cost
    local model_file=paths.concat(opt.save,"model.net")
    --os.execute('mkdir -p' .. sys.dirname(model_file))
    --torch.save(model_file,model)
    print("==> Saving model completed!\n")
    
    -- next epoch
    dataset:shuffleData()
  end
    

end
