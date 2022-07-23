// Shortening the module name
mod perceptron;
use perceptron::Perceptron;

// ----------------------------------------------------------------------- //
#[derive(Debug, Clone)]
struct MultiLayerPercetron {
    bias: f64,                  // The bias term. The same bias is used for all neurons.
    eta: f64,                   // Learning rate
    layers: Vec<usize>,         // Array with the number of elements per layer
    values: Vec<Vec<f64>>,
    d: Vec<Vec<f64>>,
    network: Vec<Vec<Perceptron>>
}

impl MultiLayerPercetron {
    fn new(layers: Vec<usize>, bias: f64, eta: f64) -> Self {
        // Initialization vector parameters
        let mut values = Vec::new();
        let mut d = Vec::new(); //  The list of lists of error terms (d = lowercase deltas)
        let mut network = Vec::new();
        // Let space for the input layer
        network.push(Vec::new());
        // Start instancianting the entinties you need to store data
        for i in 0..layers.len() {
            values.push(vec![0.0; layers[i]]);
            d.push(vec![0.0; layers[i]]);
            // instaciate the network of neurons
            if i > 0 {
                // Neuron start from the second layers, and the accept the as input the number of node predent to them plus the bias input 
                network.push(vec![Perceptron::new(layers[i - 1] , bias); layers[i]]);
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
        // Start from 1 because the first layer does not contains neurons
        for i in 1..self.layers.len() {
            for j in 0..self.layers[i]  {
                self.network[i][j].set_weights(w_init);
            }
        }
    }

    #[allow(unused)]
    fn print_weights(&self) {
        for i in 1..self.layers.len() {
            println!("Layer {}", i);
            // The second for cycle start from 1.0 because it is the second layer that contains neurons
            for j in 0..self.layers[i] {
                println!("Neuron {}, Weight {:?}", j + 1, self.network[i][j].weight)
            }
        }
    }

    fn run(&mut self, x: Vec<f64>) -> Vec<f64> {
        // Setting the first layer as the input layer
        self.values[0] = x;
        // hence from the second layer
        for i in 1..self.layers.len() {
            for j in 0..self.layers[i] {
                self.values[i][j] = self.network[i][j].run(&mut self.values[i - 1]);
            }
        }
        self.values[self.layers.len() - 1].clone()
    }

    // Upload weights based on an input vector (x) and an output vector (y)
    fn back_propagation(&mut self, x: Vec<f64>, y: Vec<f64>) -> f64 {
        let last_layer = self.layers.len() - 1;

        // Feed a sample to the NN and take the output vector for a comparison with y (the real result)
        let output = self.run(x);
        if output.len() != y.len() {
            panic!("The output vector is not the same length as the input vector")
        }

        let mut error: Vec<f64> = Vec::new();
        let mut mse: f64 = 0.0;

        // Calculate the error term of each perceptron on each layer          // =================================> the mistake is in hear, weights do not updates
        for i in (0..self.layers.len()).rev() {
            //  penultimo ... 2 1 0
            for j in 0..self.network[i].len() {
                if i == self.layers.len() - 1 {
                    // Calculate the mse - Mean Squared Error and the vector delta (the output error terms) in the last layer
                    error.push(y[j] - output[j]);
                    self.d[i][j] = output[j] * (1.0 - output[j]) * (y[j] - output[j]);
                    // println!(
                    //     "
                    //     output = {:?},
                    //     y - output = {:?},
                    //     delta = {:?}",
                    //     output,
                    //     (y[0] - output[0]),
                    //     self.d
                    // );
                } else {
                    // Instanciate the forward error
                    let mut fwd_error = 0.0;
                    for k in 0..self.network[i + 1].len() {
                        fwd_error += self.network[i + 1][k].weight[j] * self.d[i + 1][k];
                    }
                    //println!("{}", fwd_error);
                    self.d[i][j] = self.values[i][j] * (1.0 - self.values[i][j]) * fwd_error;
                }
            }
            // Evaluate the MSE
            match self.layers.last() {
                Some(n) => mse = error.iter().sum::<f64>().powf(2.0) / n.clone() as f64,
                None => panic!("Invalid layer value at the end of the NN"),
            }
        }

        // Calculate the deltas and update the weights
        for i in 1..last_layer {
            for j in 0..self.layers[i] {
                let mut delta: f64;
                for k in 0..self.layers[i - 1]  { 
                    // quantify the weight correction associated to the bias term
                    if k == self.layers[i]+1 {
                        delta = self.eta * self.d[i][j] * self.bias;
                    }
                    // quantify the weight correction  associated to the neurons
                    else {
                        delta = self.eta * self.d[i][j] * self.values[i - 1][k];
                    }
                    // updating NN weights
                    self.network[i][j].weight[k] += delta;
                }
            }
        }
        //println!("MSE = {:?}",mse);
        mse
    }
}

fn main() {
    // Inizialization of the multilayer network and uniform weights alloation
    let mut m_l = MultiLayerPercetron::new(vec![7, 3, 10], 0.0, 10.0);
    let init_weight = 0.01;
    m_l.set_weight(init_weight);

    println!("\n------------------ NN ---------------------- \n\nAll the weight are  equal to {init_weight}");
    println!(
        "The output vector should be 0 and it is equal to {:?}",
        m_l.run(Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]))
    );
    // Starting the training procedure
    let mut mse = 0.0;

    for _ in 0..2000 {
        mse = m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            Vec::from([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ); // 0 pattern
        mse = m_l.back_propagation(
            Vec::from([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Vec::from([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ); // 1 pattern
        mse = m_l.back_propagation(
            Vec::from([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
            Vec::from([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ); // 2 pattern
        mse = m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ); // 3 pattern
        mse = m_l.back_propagation(
            Vec::from([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ); // 4 pattern
        mse = m_l.back_propagation(
            Vec::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ); // 5 pattern
        mse = m_l.back_propagation(
            Vec::from([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ); // 6 pattern
        mse = m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        ); // 7 pattern
        mse = m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ); // 8 pattern
        mse = m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ); // 9 pattern
    }
    // println!("\nThe mean squared error is: {}", mse);
    //m_l.print_weights();
    println!("MSE = {}", mse);
    let see = m_l.run(Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    println!("The result of a test is  and it should be 0.8 {:?}", see);
    /*println!(
        "the values of neurons in the middel layer are: {:?}",
        m_l.values[1]
    );
    println!(
        "the values of the weight fot in the middel neurons are: {:?}",
        m_l.network[1]
    ); */
    //println!("\nAfter training, the output vector should be [0.9] and it is {:?}", see);
}
