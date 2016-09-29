local t = torch.Tensor({{1,2,3},{4,5,6},{7,8,9}})

local col = t:select(2, 2)
print(col)
col = t[{{}, 2}]
print(col)
col = t:narrow(2, 2, 1)
print(col)

-- A Tensor has dimensions, while Storage does not and is simply a range of memory
