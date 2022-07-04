// Shortening the module name
mod perceptron;
use perceptron::Perceptron;

fn run_perceptron() {
    let mut and_gate = Perceptron::new(2, -10.0);
    and_gate.set_weights(20.0);
    let result = and_gate.run(vec![0.0, 0.0]);
    println!("Outlet {:?}:", result.round());
}

// ----------------------------------------------------------------------- //

#[allow(dead_code, unused)]
struct MultiLayerPercetron {
    bias: f64,          // The bias term. The same bias is used for all neurons.
    eta: f64,           // Learning rate
    layers: Vec<usize>, // Array with the number of elements per layer
    // values: Matrix,
    // d: Matrix,
    // network: Matrix
    values: Vec<Vec<f64>>,
    d: Vec<Vec<f64>>,
    network: Vec<Vec<Perceptron>>,
}

impl MultiLayerPercetron {
    fn new(layers: Vec<usize>, bias: f64, eta: f64) -> Self {
        // Initialization vector parameters
        let mut values = Vec::new();
        let mut d = Vec::new(); //  The list of lists of error terms (d = lowercase deltas)
        let mut network = Vec::new();

        // let mut neuron = Perceptron { weight };

        for i in 0..layers.len() {
            values[i] = vec![0.0; layers[i]];
            d[i] = vec![0.0; layers[i]];
            if i > 0 {
                network[i] = vec![ Perceptron { vec![0.0, layers[i-1],  bias]} ; layers[i] ]
            }
        }
        // Initialization Multipercetron parameters
        Self {
            bias,
            eta,
            layers,
            values,
            d,
            network,
        }
    }
}

fn main() {}
