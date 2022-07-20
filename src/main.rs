// Shortening the module name
mod perceptron;
use perceptron::Perceptron;

fn run_perceptron() {
    let mut and_gate = Perceptron::new(2, -1.00);
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
            // The second for cycle start from 1.0 because it is the first layer that contains neurons 
            for j in 0..self.layers[i] {
                self.network[i][j].set_weights(w_init);
            }
        }
    }

    fn print_weights(self)  {
        for i in 1..self.layers.len() {
            // The second for cycle start from 1.0 because it is the first layer that contains neurons 
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
        self.values[self.layers.len() - 1].clone()
    }

    // Upload weights based on an input vector (x) and an output vector (y)
    fn back_propagation(&mut self, x: Vec<f64>, y: Vec<f64>) -> f64 {

        // Repetitive code sugar 
        let last_index = self.layers.len() -1;
        // Feed a sample to the NN and take the output vector for a comparison with y (the real result) 
        let output = self.run(x);
        // Calculate the mse - Mean Squared Error
        let mut error: f64 = 0.0;
        for i in 0..y.len() {
            error += y[i]-output[i];
        }   
        let mse =  error.powf(2.0) / self.layers[last_index] as f64;
        
        // Calculate the vector delta (the output error terms) in the last layer 
        for i in 0..self.layers[last_index]-1 {
            self.d[last_index][i] = output[i] * (1.0 - output[i]) * (error);
        }

        // Calculate the error term of each unit on each layer  ** REVIEW
        for i in last_index-2..1 {
            for j in 0..self.layers[i]-1  {
                // instanciate the forward error 
                let mut fwd_error = 0.0;
                for k in 0..self.layers[i-1] {
                    fwd_error += self.network[i-1][k].weight[j] * self.d[i-1][k] 
                }
                self.d[i][j] = self.values[i][j] * (1.0 - self.values[i][j]) * fwd_error
            }
        } 
        
        // Calculate the deltas and update the weights
        for i in 1..self.layers.len() -1 {
            for j in 0..self.layers[i] {
                    let mut delta = 0.0;
                for k in 0..self.layers[i-1] {
                    if k==self.layers[i-1] {
                        delta = self.eta * self.d[i][j] * self.bias;
                    } else {
                        delta = self.eta * self.d[i][j] * self.values[i-1][k];
                    } 
                    self.network[i][j].weight[k] += delta;
                }
            }
        } 
        mse
    }
}

fn  main() {
 
    let mut m_l = MultiLayerPercetron::new(vec![7,9,9,8,8,8,8,8,1], 0.0, 0.4);
    m_l.set_weight(0.0);

    println!("The output vector should be 0 and it is equal to {:?}", m_l.run(Vec::from([1.0,1.0,1.0,1.0,1.0,1.0,0.0])));

    let mut mse = 0.0;
    for i in 0..1000{
        mse += m_l.back_propagation(Vec::from([1.0,1.0,1.0,1.0,1.0,1.0,0.0]),Vec::from([0.0]));  // 0 pattern
        mse += m_l.back_propagation(Vec::from([0.0,1.0,1.0,0.0,0.0,0.0,0.0]),Vec::from([1.0]));  // 1 pattern
        mse += m_l.back_propagation(Vec::from([1.0,1.0,0.0,1.0,1.0,0.0,1.0]),Vec::from([2.0]));  // 2 pattern
        mse += m_l.back_propagation(Vec::from([1.0,1.0,1.0,1.0,0.0,0.0,1.0]),Vec::from([3.0]));  // 3 pattern
        mse += m_l.back_propagation(Vec::from([0.0,1.0,1.0,0.0,0.0,1.0,1.0]),Vec::from([4.0]));  // 4 pattern
        mse += m_l.back_propagation(Vec::from([1.0,0.0,1.0,1.0,0.0,1.0,1.0]),Vec::from([5.0]));  // 5 pattern
        mse += m_l.back_propagation(Vec::from([1.0,0.0,1.0,1.0,1.0,1.0,1.0]),Vec::from([6.0]));  // 6 pattern
        mse += m_l.back_propagation(Vec::from([1.0,1.0,1.0,0.0,0.0,0.0,0.0]),Vec::from([7.0]));  // 7 pattern
        mse += m_l.back_propagation(Vec::from([1.0,1.0,1.0,1.0,1.0,1.0,1.0]),Vec::from([8.0]));  // 8 pattern
        mse += m_l.back_propagation(Vec::from([1.0,1.0,1.0,1.0,0.0,1.0,1.0]),Vec::from([9.0]));  // 9 pattern
    }
    println!("{}",mse);

    let see = m_l.run(Vec::from([1.0,1.0,1.0,1.0,1.0,1.0,0.0]));
    println!("After training, the output vector is {:?}", see);
}
