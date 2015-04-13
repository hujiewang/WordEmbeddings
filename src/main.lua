require 'torch'
require 'BillionWords'
require 'model'
require 'DataSet'
require 'tools'
require 'train'
require 'predict'
require 'optim'
require 'xlua'


--require('mobdebug').start()

opt={
  -- Training hyperparameters
  type = 'cuda',
  optimization = 'SGD',
  learning_rate = 1e-3,
  weight_decay = 0.1,
  momentum = 0.9,
  batch_size = 1024,
  loss = 'nll ',
  max_epochs=1,

  -- Data parameters
  word_embedding_size = 100,
  context_size = 5,
  vocab_size,
  max_train_size = 1000,
  max_valid_size = 50,
  max_test_size = 50,
  
  -- Model parameters
  hidden_layer_size = 200,
  output_layer_size = 100,

  -- Prediction Parameters
  predict_batch_size = 4096,
  
  -- Logger
  save="../log/",
  
  -- Others
  seed = 1,
  threads =2,
  
  -- BillionWords
  word_map="../data/billionwords/word_map.th7",
  test_data="../data/billionwords/test_data.th7",
  valid_data="../data/billionwords/valid_data.th7",
  train_tiny="../data/billionwords/train_tiny.th7",
  train_small="../data/billionwords/train_small.th7",
  train_full="../data/billionwords/train_full.th7",
  word_tree="../data/billionwords/word_tree1.th7",
  sentence_start_id = 793470,
  sentence_end_id = 793471,
  sentence_unknown_id = 793469,
  root_id = 880542
}


--[[GPU or CPU]]--
if opt.type == 'cuda' then
   require 'cutorch'
   require 'cunnx'
   --torch.setdefaulttensortype('torch.CudaTensor')
   print('Global: switching to CUDA')
else
  require 'nn'
end

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

billionwords = BillionWords(opt)

train_dataset,valid_dataset,test_dataset = billionwords:loadData()

opt.vocab_size = #billionwords.word_map

model,criterion = getModel(opt,billionwords)

collectgarbage()
train(model,criterion,train_dataset,opt)

--Validation

accuracy = predict(valid_dataset,model,billionwords,opt)

predictSingle({"I","want","to","have","a"},model,billionwords,opt)

print("accuracy(validation): "..accuracy *100 .."%\n")
