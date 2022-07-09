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
#[derive(Debug, Clone)]
#[allow(dead_code, unused)]
struct MultiLayerPercetron {
    bias: f64,            // The bias term. The same bias is used for all neurons.
    eta: f64,             // Learning rate
    layers: Vec<usize>,   // Array with the number of elements per layer
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
        // Let space for the input layer
        network.push(Vec::new());

        for i in 0..layers.len() {
            values.push(vec![0.0; layers[i]]);
            d.push(vec![0.0; layers[i]]);

            if i > 0 {
                // Vector of one layer of the NN, since they have the same inputs they can be summirized in this way 
                network.push(vec![Perceptron::new(layers[i - 1], bias); layers[i]]);
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

    fn set_weight(&mut self, w_init: f64) {
        for i in 1..self.layers.len() {
            // The second for cycle start from 1 because it is the first layer that contains neurons 
            for j in 0..self.layers[i] {
                self.network[i][j].set_weights(w_init);
            }
        }
    }

    fn print_weights(self)  {
        for i in 1..self.layers.len() {
            // The second for cycle start from 1 because it is the first layer that contains neurons 
            for j in 0..self.layers[i] {
                println!("Layer: {}, Neuron {}, Weight {:?}",  i+1, j+1, self.network[i][j].weight )
            }
        }
    }

    fn run(&mut self, x: Vec<f64> ) -> Vec<f64> {
        self.values[0] = x;

        for i in 1..self.layers.len() {
            for j in 0..self.layers[i] {
                self.values[i][j] = self.network[i][j].run(self.values[i-1].clone());
            }
        } 
        self.values[self.layers.len() -1].clone()
    }
}

fn  main() {
    //let a =  vec![[[Perceptron::new(1,1.0)]]];
    let mut m_l = MultiLayerPercetron::new(vec![2,2,1], 0.0, 0.0);
    m_l.set_weight(5.0);
    

    let output = m_l.run(Vec::from([0.0, 1.0]));
    // m_l.print_weights();

    println!("The input vector is {:?}", m_l.values[0]);
    println!("The output vector is {:?}", output);

}
