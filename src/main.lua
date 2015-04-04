require 'torch'
require 'BillionWords'
require 'model'
require 'DataSet'
require 'train'
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
  batch_size = 512,
  loss = 'nll ',
  max_epochs=1,

  -- Data parameters
  word_embedding_size = 10,
  context_size = 3,
  vocab_size,

  -- Model parameters
  hidden_layer_size = 10,
  output_layer_size,

  -- Logger
  save="../log/",
  
  -- Others
  seed = 1,
  threads =2
}

--[[GPU or CPU]]--
if opt.type == 'cuda' then
   require 'cutorch'
   require 'cunn'
   print('==> switching to CUDA')
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)


billionwords_opt={
  word_map="../data/billionwords/word_map.th7",
  test_data="../data/billionwords/test_data.th7",
  valid_data="../data/billionwords/valid_data.th7",
  train_tiny="../data/billionwords/train_tiny.th7",
  train_small="../data/billionwords/train_small.th7",
  train_full="../data/billionwords/train_full.th7",
  word_tree="../data/billionwords/word_tree1.th7",
  context_size = 3,
  sentence_start_id = 793470,
  sentence_end_id = 793471,
  sentence_unknown_id = 793469,
  root_id = 880542
}

billionwords = BillionWords(billionwords_opt,opt)

dataset = billionwords:loadData()

--opt.output_layer_size = #billionwords.word_map
opt.output_layer_size = 10
opt.vocab_size = #billionwords.word_map

model,criterion = getModel(opt)
--io.write("Press <Enter> to continue...")
--io.read()
collectgarbage()
train(model,criterion,dataset,opt)