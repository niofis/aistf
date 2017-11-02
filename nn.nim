import
  sequtils

type Neuron = tuple[weights: seq[float]]

#type
#  Layer = ref LayerObj
#  LayerObj = object
#    neurons: seq[Neuron]
#    previous: ref Layer
#    next: ref Layer

type
  Layer = ref LayeTuple
  LayerTuple = tuple[neurons: seq[Neuron], previous: ref Layer, next: ref Layer]

type Network = tuple[layers: seq[Layer]]

proc newNeuron(weights_count: int): Neuron =
  (weights: newSeq[float](weights_count))

proc newLayer(size: int, previous: ref Layer, next: ref Layer): Layer =
  let inputs = if not previous is nil: previous.len else: 1
  
  var neurons = toSeq(size).mapIt(newNeuron(inputs))

  (neurons: neurons, previous: previous, next: next)

proc newNetwork(layers_sizes: seq[int], sigmoid_slope = 2): Network =
  var layers = layers_sizes.map(proc (size: int): auto =
      newSeq[float](size)
    )
  ()

echo newNetwork(@[3,2,1])


