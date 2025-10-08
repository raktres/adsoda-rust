#![warn(clippy::all, clippy::pedantic)]

// multiply matrices

fn multiply_matrices(m1: &Vec<Vec<f64>>, m2: &Vec<f64>) -> Vec<f64> {
    let mut result = Vec::new();
    for i in 0..m2.len() {
        let mut res = 0.0;
        for j in 0..m1[0].len() {
            res += m2[j] * m1[i][j];
        }
        result.push(res);
    }
    result
}

// flip

fn flip(equ: &Vec<f64>) -> Vec<f64> {
    equ.iter().map(|&coord| -coord).collect()
}


// Fonction `subtract`

fn Subtract(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}


// Fonction `matnorm`


fn matnorm(a: &Vec<f64>) -> f64 {
    let res: f64 = a.iter().map(|&x| x * x).sum();
    res.sqrt()
}


// Fonction `matdot`


fn matdot(vector1: &Vec<f64>, vector2: &Vec<f64>) -> f64 {
    vector1.iter().zip(vector2.iter()).map(|(x, y)| x * y).sum()
}

// Fonction `normalize`


fn normalize(hs: &Vec<f64>) -> Vec<f64> {
    let sum: f64 = hs.iter().take(hs.len() - 1).map(|&x| x * x).sum();
    let length = sum.sqrt();
    hs.iter().map(|&x| x / length).collect()
}

// Fonction `echelon`


fn echelon(matrix: &Vec<Vec<f64>>, very_small_num: f64) -> Vec<Vec<f64>> {
    let nbrows = matrix.len();
    let nbcolumns = matrix[0].len();
    let mut outmatrix = matrix.iter()
        .map(|row| row.iter().map(|&x| if x.abs() < very_small_num { 0.0 } else { x }).collect())
        .collect::<Vec<Vec<f64>>>();

    let mut lead = 0;
    for k in 0..nbrows {
        if nbcolumns <= lead {
            return outmatrix;
        }

        let mut i = k;
        while outmatrix[i][lead] == 0.0 {
            i += 1;
            if nbrows == i {
                i = k;
                lead += 1;
                if nbcolumns == lead {
                    return outmatrix;
                }
            }
        }
        let irow = outmatrix[i].clone();
        let krow = outmatrix[k].clone();
        outmatrix[i] = krow;
        outmatrix[k] = irow;

        let val = outmatrix[k][lead];
        for j in k..nbcolumns {
            let out = outmatrix[k][j] / val;
            outmatrix[k][j] = if out.abs() < very_small_num { 0.0 } else { out };
        }

        for l in 0..nbrows {
            if l == k {
                continue;
            }
            let val = outmatrix[l][lead];
            for j in k..nbcolumns {
                let nval = outmatrix[l][j] - val * outmatrix[k][j];
                outmatrix[l][j] = if nval.abs() < very_small_num { 0.0 } else { nval };
            }
        }
        lead += 1;
    }
    outmatrix
}
