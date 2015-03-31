require 'torch'

opt={
  type='cuda'
  optimization='SGD'
  learning_rate=0.1
  weight_decay=0.1
  momentum=0.9
  loss == 'nll'
  word_embedding_size=50
  context_size=11
  hidden_layer_size=1000
  output_layer_size=0
  max_epoch=10
}

data_loader=DataLoader()

dataset=data_loader.loadData()

opt.output_layer_size=dataset.vocab.size

model=nn.sequential()
model:add(nn.lookupTable(dataset.vocab.size,opt.word_embedding_size))
model:add(nn.Reshape(opt.context_size*opt.word_embedding_size))
model:add(nn.Linear(opt.context_size,opt.hidden_layer_size))
model:add(nn.Tanh())
model:add(nn.Linear(opt.hidden_layer_size,dataset.vocab.size))
model:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()
 
train(model,criterion,dataset,opt)