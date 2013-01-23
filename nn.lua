-- nn.lua
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
local math=require("math")
local os=require("os")
local tostring = tostring

local base = _G
module("nn")

Neuron={}

function Neuron:new(o)
    o=o or {}
    base.setmetatable(o,self)
    self.__index=self

    o.input_count=o.input_count or {}
    o.weights=o.weights or {}
    o.activation=o.activation

	o.output=nil

	Neuron.init(o)
    return o
end

function Neuron:init()
    for i=1,self.input_count do
        self.weights[i]=math.random()-0.5
    end
end

function Neuron:out(input)
    local j=math.min(#self.weights,#input)
    local acc=0
    for i=1,j do
        if self.weights[i] and input[i] then
            acc=acc+(self.weights[i]*input[i])
        end
    end


    if self.activation then
		self.output=self.activation.fn(acc)
        return self.output
    else
        return acc,0
    end
end

NeuralLayer={}
function NeuralLayer:new(o)
	o=o or {}
	base.setmetatable(o,self)
	self.__index=self

	o.size=o.size or 0
	o.neurons={}
	o.previous=o.previous
	o.next=o.next
	o.input_count =o.input_count or 0
	o.activation=o.activation

	o.output={}
	o.input={}

	if(o.previous) then
		o.input_count=o.previous.size
	end

	NeuralLayer.init(o)

	return o
end

function NeuralLayer:init()
	local n = nil
	for i=1,self.size do
		n = Neuron:new{input_count=self.input_count, activation=self.activation};
		self.neurons[i]=n
	end
end

function NeuralLayer:run(input)
	self.input=input
	for i,n in base.pairs(self.neurons) do
		self.output[i]=n:out(input)
	end
	if self.next then
		return self.next:run(self.output)
	else
		return self.output
	end
end

NeuralNetwork={}
function NeuralNetwork:new(o)
    o=o or {}
    base.setmetatable(o,self)
    self.__index=self

    --o.input_count=o.input_count or 0
    o.layers_sizes=o.layers_sizes or {}
	--o.output_count=o.output_count or 0
    --o.weights=o.weights or {}
    o.layers={}
	o.sigmoid_slope = o.sigmoid_slope or 1
    o.activation=CreateLogSigmoidFunction(o.sigmoid_slope)
    NeuralNetwork.init(o)

	o.input_layer=o.layers[1]
	o.output_layer=o.layers[#o.layers]

    return o
end

function NeuralNetwork:init()
    local l=nil

	self.layers={}

    math.randomseed(os.time())
	--First layer (input)
	--l=NeuralLayer:new{previous=nil,input_count=self.input_count,size=self.input_count,activation=self.activation}
	--self.layers[1]=l

	--Hidden layers
	for i,size in base.pairs(self.layers_sizes) do
		local s=self.input_count
		if l then s=l.size end
		l=NeuralLayer:new{previous=l,input_count=s,size=size,activation=self.activation}
		self.layers[i]=l
	end

	--Last layer (output)
	--l=NeuralLayer:new{previous=l,input_count=l.size,size=self.output_count,activation=self.activation}
	--self.layers[#self.layers+1]=l

	for i,layer in base.pairs(self.layers) do
		layer.next=self.layers[i+1]
	end
end

function NeuralNetwork:run(input)
    return self.layers[1]:run(input)
end

function NeuralNetwork:to_table()
	table t={}
	t.layers_sizes=self.layers_sizes
	t.input_scount=self.input_count
	t.sigmoid_slope=self.sigmoid_slope
	t.layers={}
	for i,v in base.pairs(self.layers) do
		t.layers[i]=v.weights
	end
	return t
end

function NeuralNetwork.from_table(data)
	local nn=NeuralNetwork:new{input_count=data.input_count,layers_sizes=data.layers_sizes, sigmoid_slope=data.sigmoid_slope}
	for i,v in base.pairs(nn.layers) do
		v.weights=data[i].weights
	end
	return nn
end

function CreateStepFunction(threshold)
    return function(x) return (x>threshold) and 1 or 0 end
end

function CreateLinearFunction()
    return nil
end

function CreateLogSigmoidFunction(slope)
    local s=slope or 1
	local fn=function(x) return 1/(1+math.exp(-x*s)) end
	local dr=function(x) return fn(x)*(1-fn(x)) end
    return {fn=fn, dr=dr}
end


--[[
	Train data format:
	{
		{
			input={},
			output={}
		},
		{
			input={},
			output={}
		}
	}
]]
NeuralTrainer={}
function NeuralTrainer:new(o)
    o=o or {}
    base.setmetatable(o,self)
    self.__index=self

    o.nn=o.network or {}
	o.learning_rate=o.learning_rate or 0.5
	o.train_data=o.train_data or {}

    return o
end

function NeuralTrainer:BackPropagation()
    local network_error=0

    for _,data in base.pairs(self.train_data) do
		local actual_out=self.nn:run(data.input)
		local expected_out=data.output

    	for i=1,#actual_out do
			network_error=network_error + math.pow(actual_out[i]-expected_out[i],2)/2;
    	end

		local layer=self.nn.output_layer
		for i,neuron in base.pairs(layer.neurons) do
			neuron.bp_error=neuron.output*(1-neuron.output)*(neuron.output-expected_out[i])
		end

		layer=layer.previous
		while layer do
			for i, neuron in base.pairs(layer.neurons) do
				local expected=0
				for j,m in base.pairs(layer.next.neurons) do
					expected=expected+m.weights[i]*m.bp_error
				end
				neuron.bp_error=neuron.output*(1-neuron.output)*expected
			end
			layer=layer.previous
		end

		--Weights update
		layer=self.nn.input_layer
		while layer do
			for i,neuron in base.pairs(layer.neurons) do
				for j,w in base.pairs(neuron.weights) do
					neuron.weights[j]=w-self.learning_rate*neuron.bp_error*layer.input[j]
				end
			end
			layer=layer.next
		end
	end

    return network_error
end

function NeuralTrainer:Rprop()
    local network_error=0

    for train_index,data in base.pairs(self.train_data) do
		local actual_out=self.nn:run(data.input)
		local expected_out=data.output

    	for i=1,#actual_out do
			network_error=network_error + math.pow(actual_out[i]-expected_out[i],2)/2;
    	end

		local layer=self.nn.output_layer
		for i,neuron in base.pairs(layer.neurons) do
			neuron.bp_error=neuron.output*(1-neuron.output)*(neuron.output-expected_out[i])
		end

		layer=layer.previous
		while layer do
			for i, neuron in base.pairs(layer.neurons) do
				local expected=0
				for j,m in base.pairs(layer.next.neurons) do
					expected=expected+m.weights[i]*m.bp_error
				end
				neuron.bp_error=neuron.output*(1-neuron.output)*expected
			end
			layer=layer.previous
		end
		
		--Calculate partial derivatives
		layer=self.nn.input_layer
		while layer do
			for i,neuron in base.pairs(layer.neurons) do
				neuron.pd=neuron.pd or {}
				for j,w in base.pairs(neuron.weights) do
					local pd=neuron.pd[j] or 0
					neuron.pd[j]=pd + neuron.bp_error*layer.input[j]
				end
			end
			layer=layer.next
		end
	end
	
	--Weights update
	layer=self.nn.input_layer
	while layer do
		for i,neuron in base.pairs(layer.neurons) do
			neuron.old_pd=neuron.old_pd or {}
			neuron.old_delta=neuron.old_delta or {}
			--neuron.old_pd[train_index]=neuron.old_pd[train_index] or {}
			--neuron.old_delta[train_index]=neuron.old_delta[train_index] or {}
			for j,w in base.pairs(neuron.weights) do
				local old_pd=neuron.old_pd[j] or 0
				local pd=neuron.pd[j]
				local r=old_pd*pd
				local old_delta=neuron.old_delta[j] or 0.1
				local delta=old_delta
				local np=1.2
				local nm=0.5
				if r>0 then
					delta=math.min(old_delta*np,50)
				elseif r<0 then
					delta=math.max(old_delta*nm,0.0000001)
					pd=0
				end
				--base.print(delta,old_delta,old_delta*np,old_delta*nm)
				local s=0
				if pd<0 then s=-1 
				elseif pd>0 then s=1
				end
				neuron.weights[j]=w-s*delta
				neuron.old_pd[j]=pd
				neuron.old_delta[j]=delta
				neuron.pd[j]=0 --Clear sum for next epoch
			end
		end
		layer=layer.next
	end

    return network_error
end


