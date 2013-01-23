-- example.lua
--[[
Copyright (c) 2012 Enrique CR

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
]]
local nn=require("nn")

local net=nn.NeuralNetwork:new{input_count=1,layers_sizes={4,3,1}, sigmoid_slope=2}

 --[[
for i,v in pairs(net.layers) do
    print(i,v)
    for j,w in pairs(v) do
        print(j,w)
        for k,x in pairs(w) do
            print(k,x)
        end
    end
end
]]

local train_data=
{
	{
		input={1},
		output={1/5}
	},
	{
		input={4},
		output={2/5}
	},
	{
		input={9},
		output={3/5}
	},
	{
		input={16},
		output={4/5}
	},
	{
		input={25},
		output={5/5}
	}
}
local trainer=nn.NeuralTrainer:new{network=net,learning_rate=1,train_data=train_data}
local i=0
while true do
	--local err=trainer:BackPropagation()
	local err=trainer:Rprop()
	io.stdout:write(i .. " " .. err .. "\r")
	i=i+1
	if err<0.000001 or i>1000000 then break end
end
print("")
--print(net.layers[1])
for input=1,25 do
	for i,v in pairs(net:run({input})) do
    	print( input .. ":" .. v .. "(".. v*5 ..")")
	end
end




