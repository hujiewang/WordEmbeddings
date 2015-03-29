
--[[
1.Converts plain text data into 11-grams
2.Generates a vocabulary data
]]--

-- Converts plain text data into 11-grams

require ('mobdebug').start()
require('matio')
require('tools')

vocab={}
train_input={}
train_target={}

local function readFile(fname)
  local f = assert(io.open(fname,"r"))
  local content = f:read("*all")
  f:close()
  return content
end

local function transform(word)
  if string.match(word,'[%d]*')==word then
      word="<NUMBER>"
  end
  return word
end

local function getWords(str)
  local words = {}
  -- Splits string by spaces
  for word in string.gmatch(str, "[^%s]+") do
    table.insert(words,transform(word))
  end
  return words
end

local function getSentences(str)
  local sentences={}
  pos=0
  while true do
    next_pos=str:find('[,.?!;:]',pos+1)
    if next_pos==nil then
      break
    end
    if str:sub(pos+1,next_pos-1):len()~=0 then
      table.insert(sentences,getWords(str:sub(pos+1,next_pos-1)))
    end
    pos=next_pos
  end
  return sentences
end

-- Checks if the 11-gram is valid or not
local function valid(words)
  for i,word in ipairs(words) do
    for j=1,#word-1 do
      if string.match(word,'^[%a\']*')~=word and word~="<NUMBER>" then
        return false
      end
    end
  end
  return true
end

local function getVocabulary(sentences)
  local vocab={}
  local seem={}
  for i,sentence in ipairs(sentences) do
    for j,word in ipairs(sentence) do
      if valid({word}) and seem[word]==nil then
        seem[word]=true
        table.insert(vocab,word)
      end
    end
  end
  return vocab
end

local function generateTrainingData(sentences)
  local train_input={}
  local train_target={}
  for i,sentence in ipairs(sentences) do
    local cur={}
    for j,word in ipairs(sentence) do
      table.insert(cur,word)
      if #cur == 11 and valid(cur)then
        --One correct 
        table.insert(train_input,deepcopy(cur))
        table.insert(train_target,1)
        --One fake
        local fake=deepcopy(cur)
        fake[6]=vocab[math.random(#vocab)]
        table.insert(train_input,fake)
        table.insert(train_target,0)
        table.remove(cur,1)
      end
    end
  end
  return train_input,train_target
end

local function writeFile(fname,content)
  local f = assert(io.open(fname,"w+"))
  for i,item in ipairs(content) do
    if type(item)=="table" then
      for j,sub_item in ipairs(item) do
        f:write(sub_item)
        if j~=#item then
          f:write(" ")
        end
      end
    else
      f:write(item)
    end
    f:write("\n")
  end
  f:close()
end


input_file="../data/test.txt"
train_input_file="../data/train_input.txt"
train_target_file="../data/train_target.txt"
vocab_file="../data/vocab.txt"
log_file="./log.txt"

sentences=getSentences(readFile(input_file))
vocab=getVocabulary(sentences)

train_input,train_target=generateTrainingData(sentences)

writeFile(train_input_file,train_input)
writeFile(train_target_file,train_target)
writeFile(vocab_file,vocab)



