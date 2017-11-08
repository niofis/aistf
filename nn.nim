import
  sequtils,
  random,
  math

type Neuron = object
  weights: seq[float]
  error:float       #for training purposes

type
  Layer = ref LayerTuple
  LayerTuple = object
    neurons: seq[Neuron]
    previous: Layer
    next: Layer

type Network = object
  layers: seq[Layer]

proc newNeuron(weights_count: int): Neuron =
  var weights = toSeq(1..weights_count).map(proc (_:int): auto = random(1.0) - 0.5);
  Neuron(weights: weights)

proc newLayer(size: int, previous: Layer, next: Layer): Layer =
  let inputs = if previous != nil: previous.neurons.len else: 1
  
  var neurons = toSeq(1..size).map(proc (_: int): auto = newNeuron(inputs))

  Layer(neurons: neurons, previous: previous, next: next)

proc newNetwork(layers_sizes: seq[int], sigmoid_slope = 2): Network =
  var prev: Layer = nil
  var layers = layers_sizes.map(proc (size: int): auto =
       result = newLayer(size, prev, nil)
       prev = result
    )
  for idx in 1..<layers.len:
    layers[idx - 1].next = layers[idx]

  Network(layers: layers)

proc run(neuron: Neuron, input: seq[float]): float =
  let acc = neuron.weights.zip(input).mapIt(it.a * it.b).foldr(a + b)
  return 1.0/(1.0 + exp(-acc))

proc run(layer: Layer, input: seq[float]): seq[float] =
  var output = layer.neurons.mapIt(it.run(input))
  if layer.next != nil:
    return layer.next.run(output)
  else:
    return output


proc run(network:Network, input: seq[float]): seq[float] =
  return network.layers[0].run(input)


type trainData = tuple[input: seq[float], output: seq[float]]

proc train(network: Network, data: trainData) =
  let actual = network.run(data.input)
  let expected = data.output
  let error = zip(actual, expected).mapIt(pow(it.a - it.b, 2) / 2.0).foldr(a + b)


proc main() =
  randomize()
  var nn = newNetwork(@[3,2,1])
  echo nn.layers.mapIt($it.neurons)
  echo nn.run(@[1.0, 1.1, 2.0])
main()



