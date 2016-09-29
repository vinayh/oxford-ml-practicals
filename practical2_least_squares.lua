-- Edited 2016-09-29 by vinayh
-- Trained parameters:
  -- 0.6501
  -- 1.1099
  -- 31.9807

----------------------------------------------------------------------
----------------------------------------------------------------------

require 'torch'
require 'optim'
require 'nn'

data = torch.Tensor{
   {40,  6,  4},
   {44, 10,  4},
   {46, 12,  5},
   {48, 14,  7},
   {52, 16,  9},
   {58, 18, 12},
   {60, 22, 14},
   {68, 24, 20},
   {74, 26, 21},
   {80, 32, 24}
}

model = nn.Sequential()                 -- define the container
ninputs = 2; noutputs = 1
model:add(nn.Linear(ninputs, noutputs)) -- define the only module
theta, dl_dx = model:getParameters()

input = data[{{}, {2,3}}]
y = torch.Tensor(input:size(1))
y = y:copy(data[{{}, 1}])
X = input:resize(input:size(1), input:size(2) + 1)
X[{{}, X:size(2)}] = 1

print(X)
print(y)

new_theta = torch.inverse(X:t() * X) * X:t() * y
theta:copy(new_theta)
print(theta)

dataTest = torch.Tensor{
  {6, 4},
  {10, 5},
  {14, 8}
}

print('id  approx')
for i = 1,(#dataTest)[1] do
   local myPrediction = model:forward(dataTest[i][{{1,2}}])
   print(string.format("%2d  %6.2f", i, myPrediction[1]))
end
