extern crate cnnks;
use cnnks::CnnksError;

/// This NN predicts the numberpads output, when it is shows the correct pattern the NN will output the numeber in that a human would associate to that numberpads */
#[test]
fn training_correctly() {
    let after_training = training_test().unwrap();
    let hoped_result =  Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let tollerance = 1e-1;
    for i in 0..after_training.len() {
        if (after_training[i]-hoped_result[i]).abs() > tollerance{
            panic!("The training did not work correctly")
        }
    }
}

fn training_test() -> Result<Vec<f64>, CnnksError> {
    // -------- Inizialization of the multilayer network and randoms weights alloation   ---------------------//
    let mut m_l = cnnks::MultiLayerPercetron::new(7, 10, vec![10,10], 1.0, 0.50);
    let init_weight = 0.50;
    m_l.set_weight(init_weight);

    println!("\n//------------------------------- NN -------------------------------------- \n");
    println!(
        "The output should be 8 however the output has this noise: \n {:?} \n",
        m_l.run(Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
            .clone()
    );

    // Starting the training procedure
    for i in 0..2000 {
        let mut mse = 0.0;
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            Vec::from([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))?; // 0 pattern
        mse += m_l.back_propagation(
            Vec::from([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Vec::from([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))?; // 1 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
            Vec::from([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))?; // 2 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))?; // 3 pattern
        mse += m_l.back_propagation(
            Vec::from([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))?; // 4 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))?; // 5 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))?; // 6 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))?; // 7 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))?; // 8 pattern
        mse += m_l.back_propagation(
            Vec::from([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
            Vec::from([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))?; // 9 pattern

        if i % 100 == 0 {
            println!("MSE = {}", mse);
        }
    }

    // Run the NN after the training procedure
    println!("\n------------------ After training  ---------------------- \n");
    let result_after_training = m_l.run(Vec::from([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]));
    println!("{:?}\n ", result_after_training);

    Ok(result_after_training.to_vec())
    // for i in 0..10 {
    //     if result_after_training[i] > 0.5 {
    //         println!("The trained NN recognize: {}", i)
    //     }
    // }
}
