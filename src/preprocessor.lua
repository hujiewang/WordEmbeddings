
--[[
1.Converts plain text data into 11-grams
2.Generates a vocabulary data
]]--

-- Converts plain text data into 11-grams

require('matio')

local function readFile(fname)
    local f = assert(io.open(fname,"r"))
    local content = f:read("*all")
    f:close()
    return content
end

local function getWords(str)
    local words = {}
    -- Splits string by spaces
    for word in string.gmatch(str, "[^%s]+") do
      table.insert(words,word)
    end
    return words
end

local function getVocabulary(words)
    local vocab={}
    
end

local generateTrainingData(words)
  local train_input={}
  local train_target={}
  local cur={}
  for word in words do
    if #cur == 11 and valid(cur)then
      --One correct 
       table.insert(train_input
      --One fake
      
    end
  end
end

input_data="../data/wikipedia.txt"

words=getWords(read_file(input_data))
vocab=getVocabulary(words)

