-- example.lua
local nn=require("nn")

local net=nn.NeuralNetwork:new{input_count=1,layers_sizes={4,3,1}, activation=nn.CreateLogSigmoidFunction(1)}

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
	local err=trainer:BackPropagation()
	io.stdout:write(i .. " " .. err .. "\r")
	i=i+1
	if err<0.00001 or i>1000000 then break end
end
print("")
--print(net.layers[1])
for input=1,25 do
	for i,v in pairs(net:run({input})) do
    	print( input .. ":" .. v .. "(".. v*5 ..")")
	end
end




