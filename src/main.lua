require 'torch'
require 'BillionWords'
require 'model'
require 'train'

--require('mobdebug').start()

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

billionwords_opt={
  word_map="../data/billionwords/word_map.th7",
  test_data="../data/billionwords/test_data.th7",
  valid_data="../data/billionwords/valid_data.th7",
  train_tiny="../data/billionwords/train_tiny.th7",
  train_small="../data/billionwords/train_small.th7",
  train_full="../data/billionwords/train_full.th7",
  word_tree="../data/billionwords/word_tree1.th7",
  context_size = 4,
  sentence_start_id = 793470,
  sentence_end_id = 793471,
  sentence_unknown_id = 793469,
  root_id = 880542
}

billionwords = BillionWords(billionwords_opt,opt)

dataset = billionwords:loadData()

opt.output_layer_size = #billionwords.word_map
opt.vocab_size = #billionwords.word_map

model,criterion = getModel(opt)

train(model,criterion,dataset,opt)