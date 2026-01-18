//! fonctions de base d'ADSODA
//!
//! les différentes méthodes qui servent ensuite aux faces, etc
//!

use memoize::memoize;
use serde::{Deserialize, Serialize};
// use uuid::Version;

// ============================================================================
//
// Global parameters
//
// ============================================================================

/// Un point peut être l'intersection de plusieurs faces.
///
/// Pour faciliter les calculs, il est intéressant de poser une limite haute au nombre de face à prendre en compte.
pub const MAX_FACE_PER_CORNER: usize = 16;

/// À cause des arrondis sur les calculs numériques, une même quantité peut être représenté par deux valeurs très proches.
///
/// Ce paramètre fixe l'écart en dessous duquel deux valeurs sont considérées comme identiques.
pub const VERY_SMALL_NUM: f64 = 0.00000000000001;

// ============================================================================
//
// Internal enumerations
//
// ============================================================================

/// les différentes formes d'export JSON pour un objet
#[derive(Default)]
pub enum JsonFormat {
    #[default]
    Export,
    Condensed,
    Pretty,
}

/// les deux type de réduction de dimension utilisés. Soit on projette, soit on coupe.
#[derive(Default, Clone, PartialEq, Debug,  Serialize, Deserialize)]
pub enum ProjecType {
    #[default]
    Projection,
    Cut,
}

// ============================================================================
//
// Axis
//
// ============================================================================

/// Un axe pour les projections
///
/// Pour l'instant on fait simple, on suit les axes du repère orthonormé
///
/// On pourrait aussi déterminer un repère, groupe de n axes
///
#[derive(Debug)]
pub struct Axis(pub usize);
impl Axis {
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }
}

// ============================================================================
//
// Orth Ref
//
// ============================================================================
/// Ensemble des différents axes
/// Voir dans le code si c'est utile

#[derive(Debug)]
pub struct Orthoref(pub Vec<Axis>);
impl Orthoref {
    /*
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }
    */
}

// ============================================================================
//
// Point
//
// ============================================================================

/// # Point !
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Point(pub Vec<f64>);

/// on utilise Vec pour faire un outil en dimension N
impl Point {
    /// Méthode pour afficher les données
    pub fn display(&self) {
        println!("point {:?}", &self.0);
    }
    /// teste si deux points sont égaux
    pub fn is_equal(&self, point: &Point) -> bool {
        vector_is_equal(&self.0, &point.0, VERY_SMALL_NUM)
    }
    /// return Point as Vec<f64>
    pub fn to_vec(&self) -> Vec<f64> {
        self.0.clone()
    }
    pub fn halfspace_position(&self, equ: &Vec<f64>) -> f64 {
        let mut p: Vec<f64> = self.0.clone();
        p.push(1.0); // on ajoute 1 derriere le point
        vector_dot(&equ, &p)
    }
    pub fn is_inside_or_on_halfspaces(&self, eqs: &Vec<Vec<f64>>) -> bool {
        eqs.iter()
            .all(|equ| self.halfspace_position(&equ) > -VERY_SMALL_NUM)
    }
}

#[cfg(test)]
mod tests_point {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    pub fn point_validation() {
        let point1 = Point(vec![1.0, 2.0, 3.0]);
        let point2 = Point(vec![1.0, 2.0, 3.0 + VERY_SMALL_NUM * 2.0]);
        let point3 = Point(vec![1.0, 2.0, 3.0 + VERY_SMALL_NUM / 2.0]);

        assert_eq!(point1.is_equal(&point2), false);
        assert_eq!(point1.is_equal(&point3), true);
    }

    // --- Tests pour Point::display ---
    #[test]
    fn test_display() {
        // Vérifie que la méthode ne panique pas et affiche quelque chose
        let point = Point(vec![1.0, 2.0, 3.0]);
        point.display(); // Impossible de tester l'affichage directement, mais on vérifie qu'il ne panique pas
    }

    // --- Tests pour Point::is_equal ---
    #[test]
    fn test_is_equal() {
        // Points égaux
        let p1 = Point(vec![1.0, 2.0, 3.0]);
        let p2 = Point(vec![1.0, 2.0, 3.0]);
        assert!(p1.is_equal(&p2));

        // Points différents
        let p1 = Point(vec![1.0, 2.0, 3.0]);
        let p2 = Point(vec![1.0, 2.0, 3.1]);
        assert!(!p1.is_equal(&p2));

        // Points égaux avec petite différence
        let p1 = Point(vec![1.0, 2.0, 3.0]);
        let p2 = Point(vec![1.0, 2.0, 3.0 + VERY_SMALL_NUM / 2.0]);
        assert!(p1.is_equal(&p2));

        // Points de dimensions différentes
        let p1 = Point(vec![1.0, 2.0]);
        let p2 = Point(vec![1.0, 2.0, 3.0]);
        assert!(!p1.is_equal(&p2));
    }

    // --- Tests pour Point::to_vec ---
    #[test]
    fn test_to_vec() {
        // Cas normal
        let point = Point(vec![1.0, 2.0, 3.0]);
        assert_eq!(point.to_vec(), vec![1.0, 2.0, 3.0]);

        // Vecteur vide
        // let point = Point(vec![]);
        // assert_eq!(point.to_vec(), vec![]);
    }

    // --- Tests pour Point::halfspace_position ---
    #[test]
    fn test_halfspace_position() {
        // Cas normal : équation de plan 2D (ax + by + c = 0)
        let point = Point(vec![1.0, 2.0]);
        let equ = vec![1.0, -1.0, -2.0]; // x - y - 2 = 0
        let result = point.halfspace_position(&equ);
        assert_abs_diff_eq!(result, 1.0 - 2.0 - 2.0, epsilon = 1e-10);

        // Point en 3D
        let point = Point(vec![1.0, 2.0, 3.0]);
        let equ = vec![1.0, 1.0, 1.0, -6.0]; // x + y + z - 6 = 0
        let result = point.halfspace_position(&equ);
        assert_abs_diff_eq!(result, 1.0 + 2.0 + 3.0 - 6.0, epsilon = 1e-10);
    }

    // --- Tests pour Point::is_inside_or_on_halfspaces ---
    #[test]
    fn test_is_inside_or_on_halfspaces() {
        // Point à l'intérieur de tous les demi-espaces
        let point = Point(vec![1.0, 1.0]);
        let eqs = vec![
            vec![1.0, 0.0, -0.5], // x >= 0.5
            vec![0.0, 1.0, -0.5], // y >= 0.5
        ];
        assert!(point.is_inside_or_on_halfspaces(&eqs));

        // Point à la frontière d'un demi-espace
        let point = Point(vec![0.5, 1.0]);
        let eqs = vec![
            vec![1.0, 0.0, -0.5], // x >= 0.5
            vec![0.0, 1.0, -0.5], // y >= 0.5
        ];
        assert!(point.is_inside_or_on_halfspaces(&eqs));

        // Point à l'extérieur d'un demi-espace
        let point = Point(vec![0.4, 1.0]);
        let eqs = vec![
            vec![1.0, 0.0, -0.5], // x >= 0.5
            vec![0.0, 1.0, -0.5], // y >= 0.5
        ];
        assert!(!point.is_inside_or_on_halfspaces(&eqs));

        // Liste d'équations vide
        let point = Point(vec![1.0, 1.0]);
        let eqs = vec![];
        assert!(point.is_inside_or_on_halfspaces(&eqs));

        // Point en 3D
        let point = Point(vec![1.0, 1.0, 1.0]);
        let eqs = vec![
            vec![1.0, 0.0, 0.0, -0.5], // x >= 0.5
            vec![0.0, 1.0, 0.0, -0.5], // y >= 0.5
            vec![0.0, 0.0, 1.0, -0.5], // z >= 0.5
        ];
        assert!(point.is_inside_or_on_halfspaces(&eqs));
    }

    /*     // --- Test de panique pour halfspace_position (si équation trop courte) ---
    #[test]
    #[should_panic]
    fn test_halfspace_position_panic() {
        let point = Point(vec![1.0, 2.0, 3.0]);
        let equ = vec![1.0, 1.0]; // Trop courte pour un point 3D + 1
        let _ = point.halfspace_position(&equ);
    } */
}

// ============================================================================
//
// Halfspace
//
// ============================================================================

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Halfspace(pub Vec<f64>);

impl Halfspace {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn dimension(&self) -> usize {
        self.len().saturating_sub(1)
    }
    pub fn to_vec(&self) -> Vec<f64> {
        self.0.clone()
    }
    pub fn flip(&mut self) {
        for coord in &mut self.0 {
            *coord *= -1.0;
        }
    }
    pub fn fact(&mut self, factor: f64) {
        if let Some(last_index) = self.0.len().checked_sub(1) {
            for i in 0..last_index {
                self.0[i] *= factor;
            }
        }
    }


    /// récupère un coefficient
    ///
    /// normalement il faudrait vérifier que l'on ne récupère pas le dernier car c'est la position du plan et non sa direction
    pub fn get_coordinate(&self, axis: &Axis) -> f64 {
        self.0[axis.0]
    }
    /// la constante est enregistrée dans la dernière valeur du vecteur
    ///
    pub fn get_constant(&self) -> f64 {
        self.0[self.dimension()]
    }
    pub fn normalize(&mut self) {
        self.0 = vector_normalize(&self.0);
    }

    /// ajoute une constante, revient à décaler l'hyperplan selon son axe
    ///
    /*
    pub fn halfspace_add_constant(halfspace: &mut Vec<f64>, x: f64) {
        let l = halfspace.len();
        halfspace[l - 1] -= x;
    }
    */

    /// indique si un point est dans le halfspace ou non
    ///
    /// l'information est fonction du signe du résultat
    pub fn position_point(&self, point: &Point) -> f64 {
        let mut p: Vec<f64> = point.0.clone();
        p.push(1.0); // on ajoute 1 derriere le point
        vector_dot(&self.to_vec(), &p)
    }

    // enlever diff ? directement utiliser very_small ?
    pub fn is_equal(&self, hp2: &Halfspace, diff: f64) -> bool {
        // TODO
        // est-ce qu'il y a besoin de vérifier à chaque fois ??
        // let hp1n = vector_normalize(&self.to_vec());
        // let hp2n = vector_normalize(&hp2.to_vec());

        // println!("{:?} {:?} ", hp1n, hp2n);

        //  is_vector_equal(&hp1n, &hp2n, diff)
        vector_is_equal(&self.to_vec(), &hp2.to_vec(), diff)
    }

    pub fn translate(&mut self, vector: &Vec<f64>) {
        let last = self.dimension();
        let mut hsvect = self.to_vec();
        hsvect.remove(last);
        hsvect.push(self.0[last] - vector_dot(&hsvect, &vector));
        self.0 = vector_normalize(&hsvect);
        // self.0[last] -= vector_dot(&hsvect, &vector);
    }

    pub fn transform(&mut self, matrix: &Vec<Vec<f64>>) {
        // normalement m est plus petit que v de 1

        let equ = self.to_vec();
        let last = self.dimension();
        let mut halfspace = self.to_vec();
        halfspace.remove(last);

        //  The normal of the tranformed halfspace can be found with a simple matrix
        //  multiplication.

        let mut coords: Vec<f64> = matrix_multiply_v(&matrix, &equ);

        // The constant is more difficult.  Here I have solved this
        //  by finding a point on the original halfspace (by checking axes for intersections)
        //  and transforming that point as well.  The transformed point lies on the
        //  transformed halfspace, so the constant term can be computed by plugging the
        //  transformed point into the equation of the transformed halfspace (the coefficients
        //  being the coordinates of the transformed normal and the constant unknown) and
        //  solving for the constant.

        let coordindex = equ
            .iter()
            .position(|&r| r.abs() > VERY_SMALL_NUM)
            .unwrap_or(0);

        /*
        #[cfg(test)]
        println!("coord index {:?} face equ {:?}", coordindex, self.halfspace);

        #[cfg(test)]
        println!(
            "constante {:?} val coord {:?}",
            self.halfspace.get_constant(),
            self.halfspace.get_coordinate(&Axis(coordindex))
        );
        */

            // ???? -self ?
        let intercept = self.get_constant() / self.get_coordinate(&Axis(coordindex));

        //  At this point we have found a point on the halfspace.  This point is
        //  (0, 0, ..., intercept, ..., 0, 0), where intercept is the ith coordinate
        //  and all other coordinates are 0.  Since this is a highly sparse and
        //  predictable vector.  We will NOT actually plug all these coordinates
        //  into a Vector and use matrix multiplication; rather, we will take
        //  advantage of the fact that multiplication by such a vector yields
        //  a vector which is simply the ith column of m multiplied by intercept.
        //  We skip another step by plugging the coordinates of this product
        //  directly into the transformed equation.
        //
        let mut sum = 0.0;
        for i in 0..coords.len() {
            sum += matrix[i][coordindex] * intercept * coords[i]
        }
        // ???? -sum ???
        coords.push(sum);
        self.0 = vector_normalize(&coords);
    }
}


#[cfg(test)]
mod tests_halfspace {
    // use super::*;
    // use approx::assert_abs_diff_eq;

}

// ============================================================================
//
// Static
//
// ============================================================================

/// calcule l'intersection de plusieurs hyperplans
/// retourne None s'il n'y a pas de résultat
/// nota, untiliser unwrap pour l'extraire

pub fn hyperplanes_intersect(hyperplanes: &Vec<Vec<f64>>) -> Option<Vec<f64>> {
    let dimension = hyperplanes[0].len() - 1;
    if let Some(result) = matrix_solution(hyperplanes) {
        if result.len() == dimension {
            return Some(result);
        }
    }
    None
}

/// on donne un ensemble de point, on considère que ces points sont dans un hyperplan
/// et on calcule la normale à cet hyperplan
pub fn halfspaces_findnormal(points_array: &Vec<Vec<f64>>) -> Option<Vec<f64>> {
    let mut mat = points_array.clone();
    for row in &mut mat {
        row.push(1.0);
    }

    let ech = matrix_echelon(&mat);

    let isnull = ech.iter().enumerate().any(|(idx, row)| row[idx] == 0.0);
    if isnull {
        return None;
    }

    let res: Vec<f64> = ech.iter().map(|row| row[row.len() - 1]).collect();
    let mut normal = res.clone();
    // ???? -vector ???
    normal.push(vector_dot(&res, &points_array[0]));

    Some(normal)
}

#[test]
fn findnormal_validation() {
    let points_array1 = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    let points_array2 = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    let normal1 = vec![1.0, 1.0, 1.0, -1.0];

    assert_eq!(halfspaces_findnormal(&points_array1), None);
    //let result = some_function_call(some_parameters).ok_or("Some error")?;
    assert_eq!(halfspaces_findnormal(&points_array2).unwrap(), normal1);
}

/// # generate_combinations
/// génère les différentes combinaisons pour n valeurs
///
/// memoize est essentiel pour garder des performances acceptables
/// en effet l'algorithme doit essayer toutes les combinaisons croisées
/// conserver en mémoire les liens entre combinaison permet un
/// petit gain de performance sur ce problème structurel
#[memoize(SharedCache)]
fn generate_combinations(
    l: usize,
    a: usize,
    b: usize,
    start: usize,
    current: Vec<usize>,
) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    if !current.is_empty() && current.len() >= a && current.len() <= b {
        result.push(current.clone());
    }
    if current.len() < b {
        for i in start..l {
            let mut new_current = current.clone();
            new_current.push(i);
            result.extend(generate_combinations(l, a, b, i + 1, new_current));
        }
    }
    result
}

/// TODO: on va essayer de mettre cette fonction en memoize
/// pour un gain sur le tri
/// il serait bon de faire des tests de performance avec et sans
#[memoize(SharedCache)]
pub fn among_index(l: usize, a: usize, b: usize) -> Vec<Vec<usize>> {
    let mut result = generate_combinations(l, a, b, 0, vec![]);
    result.sort_by(|a, b| b.len().cmp(&a.len()));
    result
}


#[cfg(test)]
mod tests_static {
    use super::*;
    // use approx::assert_abs_diff_eq;
#[test]
fn base_validation() {
    let res = among_index(4, 2, 3);
    println!("{res:?}");
    assert_eq!(res.len(), 10);
}
}

// ============================================================================
//
// Vector
//
// ============================================================================

/// # vector_normalize
/// normalise un vecteur en divisant chaque coordonnée par la norme
///
/// attention, dans le calcul de la norme ne prend pas la dernière valeur qui correspond à la position
///
/// par contre on divise bien cette valeur par la norme
pub fn vector_normalize(v: &Vec<f64>) -> Vec<f64> {
    let sum: f64 = v
        .iter()
        .take(v.len().saturating_sub(1))
        .map(|&x| x * x)
        .sum();
    let norm = f64::sqrt(sum);
    // racine carré de la somme des carrés

    // retourne un vecteur nul si la norme est nulle
    if norm == 0.0 {
        return vec![0.0; v.len()];
    }
    v.iter().map(|&x| x / norm).collect()
}

/// # vector_dot
/// calcule le produit scalaire de deux vecteurs
pub fn vector_dot(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
}

/// retourne un vecteur opposé
///

pub fn vector_flip(v: &Vec<f64>) -> Vec<f64> {
    let mut w = v.clone();
    for coord in &mut w {
        *coord *= -1.0;
    }
    w
}

/// # vector_subtract
/// soustraction de deux vecteurs
///
/// On suppose que les 2 vecteurs ont la même taille
pub fn vector_subtract(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// # Remarque
/// c'est une simplification excessive de cette implémentation d'Adsoda
/// on ne projette que selon les axes du repère orthonormé
/// pour faire une projection il suffit donc de l'index de l'axe
/// et de supprimer cette valeur dans le vecteur
/// # TODO:
/// il pourrait être pertinent de projeter selon un vecteur donné
/// mais un peu plus complexe à définir, plus long à calculer, et il faut
/// revoir l'interface visuelle
pub fn vector_project(vector: &Vec<f64>, axis: &Axis) -> Vec<f64> {
    let mut v = vector.clone();
    v.remove(axis.0);
    v
}

/// définit le vecteur qui permet de transformer corner1 en corner2
///
/// pas sur que ce soit utilisé
/*
fn vector_from_points(corner1: &Vec<f64>, corner2: &Vec<f64>) -> Vec<f64> {
    vector_subtract(corner1, corner2)
}
    */

/// compare deux points et indique s'ils sont égaux ou non
///
/// on indique l'ecart minimal acceptable
///
/// intégré à point !!
// enlever diff ? directement utiliser very_small ?
pub fn vector_is_equal(vect1: &Vec<f64>, vect2: &Vec<f64>, diff: f64) -> bool {
    // abs_diff_eq!(vect1, vect2, epsilon = VERY_SMALL_NUM)

    if vect1.len() != vect2.len() {
        return false;
    }
    for (c1, c2) in vect1.iter().zip(vect2.iter()) {
        if (c1 - c2).abs() > diff {
            return false;
        }
    }
    true
}

/// ce serait bien de faire des tests aux limites, de faire des tests d'erreur
#[cfg(test)]
mod test_vector {
    use super::*;
    use approx::assert_abs_diff_eq;
    // ==============

    // --- Tests pour vector_normalize ---
    #[test]
    fn test_vector_normalize() {
        // Cas normal
        let v = vec![3.0, 4.0, 10.0];
        let norm = vector_normalize(&v);
        let expected_norm = f64::sqrt(3.0 * 3.0 + 4.0 * 4.0);
        assert_abs_diff_eq!(norm[0], 3.0 / expected_norm, epsilon = 1e-10);
        assert_abs_diff_eq!(norm[1], 4.0 / expected_norm, epsilon = 1e-10);
        assert_abs_diff_eq!(norm[2], 10.0 / expected_norm, epsilon = 1e-10);

        // Cas limite : vecteur de taille 1
        let v = vec![5.0];
        let norm = vector_normalize(&v);
        assert_eq!(norm, vec![0.0]);
        //  assert!(norm[0].is_nan()); // division par zéro

        // cas norme nulle, retourne vecteur nul
        let v = vec![0.0, 0.0, 10.0];
        let norm = vector_normalize(&v);
        assert_eq!(norm, vec![0.0, 0.0, 0.0]);

        // Valeurs négatives
        let v = vec![-3.0, -4.0, 10.0];
        let norm = vector_normalize(&v);
        let expected_norm = f64::sqrt(9.0 + 16.0);
        assert_abs_diff_eq!(norm[0], -3.0 / expected_norm, epsilon = 1e-10);
    }

    // --- Tests pour vector_dot ---
    #[test]
    fn test_vector_dot() {
        // Cas normal
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![4.0, 5.0, 6.0];
        assert_abs_diff_eq!(vector_dot(&v1, &v2), 32.0, epsilon = 1e-10);

        // Vecteurs nuls
        let v1 = vec![0.0, 0.0, 0.0];
        let v2 = vec![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(vector_dot(&v1, &v2), 0.0, epsilon = 1e-10);

        // Valeurs négatives
        let v1 = vec![-1.0, -2.0, -3.0];
        let v2 = vec![1.0, 1.0, 1.0];
        assert_abs_diff_eq!(vector_dot(&v1, &v2), -6.0, epsilon = 1e-10);
    }
    /*
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_vector_dot_panic_different_lengths() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![1.0, 2.0, 3.0];
        let _ = vector_dot(&v1, &v2);
    } */

    // --- Tests pour vector_flip ---
    #[test]
    fn test_vector_flip() {
        // Cas normal
        let v = vec![1.0, -2.0, 3.0];
        let flipped = vector_flip(&v);
        assert_eq!(flipped, vec![-1.0, 2.0, -3.0]);

        // Vecteur vide
        /*
        let v = vec![];
        let flipped = vector_flip(&v);
        assert_eq!(flipped, vec![]);
        */

        // Vecteur nul
        let v = vec![0.0, 0.0, 0.0];
        let flipped = vector_flip(&v);
        assert_eq!(flipped, vec![0.0, 0.0, 0.0]);
    }

    // --- Tests pour vector_subtract ---
    #[test]
    fn test_vector_subtract() {
        // Cas normal
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![0.5, 1.0, 1.5];
        assert_eq!(vector_subtract(&v1, &v2), vec![0.5, 1.0, 1.5]);

        // Vecteur nul
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![0.0, 0.0, 0.0];
        assert_eq!(vector_subtract(&v1, &v2), v1);
    }

    /* #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_vector_subtract_panic_different_lengths() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![1.0, 2.0, 3.0];
        let _ = vector_subtract(&v1, &v2);
    } */

    // --- Tests pour vector_project ---
    #[test]
    fn test_vector_project() {
        // Cas normal
        let v = vec![1.0, 2.0, 3.0];
        let axis = Axis(1);
        assert_eq!(vector_project(&v, &axis), vec![1.0, 3.0]);

        /* // Vecteur vide
        let v = vec![];
        let axis = Axis(0);
        assert_eq!(vector_project(&v, &axis), vec![]); */
    }

    /* #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_vector_project_panic_invalid_axis() {
        let v = vec![1.0, 2.0];
        let axis = Axis(5);
        let _ = vector_project(&v, &axis);
    } */

    // --- Tests pour vector_is_equal ---
    #[test]
    fn test_vector_is_equal() {
        // Vecteurs égaux
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.0, 2.0, 3.0];
        assert!(vector_is_equal(&v1, &v2, 1e-10));

        // Vecteurs différents
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.0, 2.0, 3.1];
        assert!(!vector_is_equal(&v1, &v2, 0.05));

        // Tailles différentes
        let v1 = vec![1.0, 2.0];
        let v2 = vec![1.0, 2.0, 3.0];
        assert!(!vector_is_equal(&v1, &v2, 1e-10));

        // Différence dans la marge
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.0, 2.0, 3.0000001];
        assert!(vector_is_equal(&v1, &v2, 1e-6));
    }

    //======================= */
}

// ============================================================================
//
// Matrix
//
// ============================================================================

/// # matrix_multiply
///
/// multiplication de deux matrices
///
/// on suppose que les 2 matrices sont de même taille
/*
fn matrix_multiply(m1: &Vec<Vec<f64>>, m2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut result = Vec::new();
    for i in 0..m1.len() {
        let mut row = Vec::new();
        for j in 0..m1[i].len() {
            row.push(m1[i][j] * m2[i][j]);
        }
        result.push(row);
    }
    result
}
*/

/// # matrix_multiply_v
///
/// multiplication matrice - vecteur
///
/// on suppose que les tailles sont compatibles
fn matrix_multiply_v(m: &Vec<Vec<f64>>, v: &Vec<f64>) -> Vec<f64> {
    let mut result = Vec::<f64>::new();
    for i in 0..m.len() {
        let mut res = 0.0;
        for j in 0..m[0].len() {
            res += m[i][j] * v[j];
        }
        result.push(res);
    }
    result
}

/// # matrix_echelon
///
/// transforme un système linéaire en matrice triangulaire
///
/// étape pour résoudre un système linéaire
/// attention, consomme la matrice
fn matrix_echelon(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let nbrows = matrix.len();
    let nbcolumns = matrix[0].len();
    let mut outmatrix = matrix
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|x| {
                    if x.abs() < VERY_SMALL_NUM {
                        0.0
                    } else {
                        x.clone()
                    }
                })
                .collect()
        })
        .collect::<Vec<Vec<f64>>>();
    // dbg!(outmatrix);
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

        outmatrix.swap(i, k);
        // dbg!(outmatrix);

        let val = outmatrix[k][lead];
        for j in k..nbcolumns {
            let out = outmatrix[k][j] / val;
            outmatrix[k][j] = if out.abs() < VERY_SMALL_NUM { 0.0 } else { out };
        }

        for l in 0..nbrows {
            if l == k {
                continue;
            }
            let val = outmatrix[l][lead];
            for j in k..nbcolumns {
                let nval = outmatrix[l][j] - val * outmatrix[k][j];
                outmatrix[l][j] = if nval.abs() < VERY_SMALL_NUM {
                    0.0
                } else {
                    nval
                };
            }
        }
        lead += 1;
    }
    // println!("résultat {:?} ", outmatrix);
    outmatrix
}

/// # matrix_solution
///
/// résout un système linéaire
/// TODO: plutot que de faire un clone, faire un emprunt
fn matrix_solution(matrix: &Vec<Vec<f64>>) -> Option<Vec<f64>> {
    let mat1 = matrix_echelon(&matrix);
    if mat1.is_empty() {
        // println!("Pas de solution");
        return None;
    }
    if mat1.len() < mat1[0].len() - 1 {
        // println!("Pas de solution");
        return None;
    }
    // si un élément nul sur la diagonale
    for index in 0..mat1[0].len() - 1 {
        if mat1[index][index] == 0.0 {
            // println!("Pas de solution");
            return None;
        }
    }
    let last: Vec<f64> = mat1.iter().map(|row| -row[row.len() - 1]).collect();
    //println!("résultat {:?} ", last);
    Some(last)
}

fn multiply_matrices(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut result = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiply_matrices_normal_case() {
        let a = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let b = vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let expected = vec![
            vec![19.0, 22.0],
            vec![43.0, 50.0],
        ];
        assert_eq!(multiply_matrices(&a, &b), expected);
    }

    #[test]
    fn test_multiply_matrices_identity() {
        let a = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let b = vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        assert_eq!(multiply_matrices(&a, &b), b);
    }

    #[test]
    fn test_multiply_matrices_zero() {
        let a = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        let b = vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let expected = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        assert_eq!(multiply_matrices(&a, &b), expected);
    }

    #[test]
    fn test_multiply_matrices_large_values() {
        let a = vec![
            vec![1e6, 2e6],
            vec![3e6, 4e6],
        ];
        let b = vec![
            vec![5e6, 6e6],
            vec![7e6, 8e6],
        ];
        let expected = vec![
            vec![19e12, 22e12],
            vec![43e12, 50e12],
        ];
        assert_eq!(multiply_matrices(&a, &b), expected);
    }

    #[test]
    fn test_multiply_matrices_1x1() {
        let a = vec![vec![2.0]];
        let b = vec![vec![3.0]];
        let expected = vec![vec![6.0]];
        assert_eq!(multiply_matrices(&a, &b), expected);
    }

    #[test]
    fn test_multiply_matrices_3x3() {
        let a = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let b = vec![
            vec![9.0, 8.0, 7.0],
            vec![6.0, 5.0, 4.0],
            vec![3.0, 2.0, 1.0],
        ];
        let expected = vec![
            vec![30.0, 24.0, 18.0],
            vec![84.0, 69.0, 54.0],
            vec![138.0, 114.0, 90.0],
        ];
        assert_eq!(multiply_matrices(&a, &b), expected);
    }
}
// ============================================================================
//
// Tests
//
// ============================================================================

#[cfg(test)]
mod tests_matrix {
    use super::*;
    use approx::assert_abs_diff_eq;
    #[test]
    fn matrix_validation() {
        //let v1 = vec![1.0, 2.0, 3.0];
        // let v2 = vec![-1.0, 1.0, 3.0];

        let _m1 = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];
        let _m2 = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];

        let _r4 = vec![
            vec![1.0, 4.0, 9.0, 16.0],
            vec![25.0, 36.0, 49.0, 64.0],
            vec![81.0, 100.0, 121.0, 144.0],
        ];

        let m6 = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let r6 = vec![vec![1.0, 0.0, -1.0], vec![0.0, 1.0, 2.0]];

        let m7 = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![0.0, 10.0, 11.0, 12.0],
        ];
        let r7 = vec![0.0, 1.0, -2.0];

        let m8 = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];

        //   assert_eq!(matrix_multiply(&m1, &m2), r4);
        assert_eq!(matrix_echelon(&m6), r6);
        assert_eq!(matrix_solution(&m7), Some(r7));
        assert_eq!(matrix_solution(&m8), None);

        // *******
    }

 
    /* fn test_matrix_multiply() {
        // Cas normal 2x2
        let m1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let m2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let result = matrix_multiply(&m1, &m2);
        assert_eq!(result, vec![vec![5.0, 12.0], vec![21.0, 32.0]]);

        // Exemple 4x4
        let m1 = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0, 16.0],
        ];
        let m2 = vec![
            vec![16.0, 15.0, 14.0, 13.0],
            vec![12.0, 11.0, 10.0, 9.0],
            vec![8.0, 7.0, 6.0, 5.0],
            vec![4.0, 3.0, 2.0, 1.0],
        ];
        let result = matrix_multiply(&m1, &m2);
        assert_eq!(
            result,
            vec![
                vec![1.0 * 16.0, 2.0 * 15.0, 3.0 * 14.0, 4.0 * 13.0],
                vec![5.0 * 16.0, 6.0 * 15.0, 7.0 * 14.0, 8.0 * 13.0],
                vec![9.0 * 16.0, 10.0 * 15.0, 11.0 * 14.0, 12.0 * 13.0],
                vec![13.0 * 16.0, 14.0 * 15.0, 15.0 * 14.0, 16.0 * 13.0],
            ]
        );

        // Matrices 1x1
        let m1 = vec![vec![2.0]];
        let m2 = vec![vec![3.0]];
        let result = matrix_multiply(&m1, &m2);
        assert_eq!(result, vec![vec![6.0]]);
    }
     */
    // --- Tests pour matrix_multiply_v ---
    #[test]
    fn test_matrix_multiply_v() {
        // Cas normal 2x2
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let v = vec![5.0, 6.0];
        let result = matrix_multiply_v(&m, &v);
        assert_eq!(result, vec![17.0, 39.0]);

        // Exemple 4x4
        let m = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0, 16.0],
        ];
        let v = vec![1.0, -1.0, 2.0, -2.0];
        let result = matrix_multiply_v(&m, &v);
        assert_eq!(
            result,
            vec![
                1.0 * 1.0 + 2.0 * (-1.0) + 3.0 * 2.0 + 4.0 * (-2.0),
                5.0 * 1.0 + 6.0 * (-1.0) + 7.0 * 2.0 + 8.0 * (-2.0),
                9.0 * 1.0 + 10.0 * (-1.0) + 11.0 * 2.0 + 12.0 * (-2.0),
                13.0 * 1.0 + 14.0 * (-1.0) + 15.0 * 2.0 + 16.0 * (-2.0),
            ]
        );

        // Vecteur nul
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let v = vec![0.0, 0.0];
        let result = matrix_multiply_v(&m, &v);
        assert_eq!(result, vec![0.0, 0.0]);
    }

    // --- Tests pour matrix_echelon ---
    #[test]
    fn test_matrix_echelon() {
        // Cas normal 3x3
        let matrix = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![0.0, 0.0, 1.0, 2.0],
        ];
        let result = matrix_echelon(&matrix);
        assert_eq!(result.len(), 3);
        assert_abs_diff_eq!(result[0][0], 1.0, epsilon = 1e-10);

        // Exemple 4x4
        let matrix = vec![
            vec![2.0, 1.0, -1.0, 8.0, 11.0],
            vec![-3.0, -1.0, 2.0, -11.0, -17.0],
            vec![-2.0, 1.0, 2.0, -3.0, -7.0],
            vec![1.0, 1.0, 1.0, 1.0, 3.0],
        ];
        let result = matrix_echelon(&matrix);
        // Vérifie que la matrice est triangulaire supérieure
        for i in 0..4 {
            for j in 0..i {
                assert_abs_diff_eq!(result[i][j], 0.0, epsilon = 1e-10);
            }
        }

        // Matrice déjà triangulaire
        let matrix = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![0.0, 0.0, 1.0, 2.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let matres = vec![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let result = matrix_echelon(&matrix);
        assert_eq!(result, matres);
    }

    // --- Tests pour matrix_solution ---
    #[test]
    fn test_matrix_solution() {
        // Cas avec solution 2x2
        let matrix = vec![vec![1.0, 0.0, 5.0], vec![0.0, 1.0, 6.0]];
        let result = matrix_solution(&matrix);
        assert!(result.is_some());
        assert_abs_diff_eq!(result.clone().unwrap()[0], -5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.unwrap()[1], -6.0, epsilon = 1e-10);

        // Exemple 4x4 avec solution
        let matrix = vec![
            vec![1.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 1.0, 0.0, 3.0],
            vec![0.0, 0.0, 0.0, 1.0, 4.0],
        ];
        let result = matrix_solution(&matrix);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec![-1.0, -2.0, -3.0, -4.0]);

        // Pas de solution (diagonale nulle)
        let matrix = vec![vec![0.0, 0.0, 5.0], vec![0.0, 0.0, 6.0]];
        let result = matrix_solution(&matrix);
        assert!(result.is_none());
    }

    #[test]
    fn halfspace_mvt_validation() {
        let halfspace1: Halfspace = Halfspace(vec![1.0, 2.0, 3.0, 4.0]);
        // let halfspace2:Vec<f64> = vec![1.0, 2.0, 3.0, 14.0];

        let point = Point(vec![3.0, 2.0, 6.0]);

        // halfspace_add_constant(halfspace1,10.0);

        // halfspace_add_constant(&mut halfspace1, -10.0);
        // assert_eq!(halfspace1, halfspace2);

        assert_eq!(halfspace1.get_constant(), 4.0);
        assert_eq!(halfspace1.get_coordinate(&Axis(0)), 1.0);

        // TODO: verifier !!!!
        assert_eq!(halfspace1.position_point(&point), 29.0);

        let point1 = vec![1.0, 2.0, 3.0];
        let point2 = vec![1.0, 2.0, 10.0];
        let point3 = vec![1.0, 2.0, 3.0 + 0.05];

        assert_eq!(vector_is_equal(&point1, &point2, 2.0), false);
        assert_eq!(vector_is_equal(&point1, &point3, 0.1), true);

        let mut hs1 = Halfspace(vec![1.0, 2.0, 3.0]);
        let mut hs2 = Halfspace(vec![2.0, 4.0, 6.2]);

        hs1.normalize();
        hs2.normalize();

        assert_eq!(hs1.is_equal(&hs2, 0.1), true);
        assert_eq!(hs1.is_equal(&hs2, 0.001), false);


        let mut halfspace7: Halfspace = Halfspace(vec![1.0, 2.0, 3.0, 4.0]);
        let halfspace8= Halfspace(vec![2.0, 4.0, 6.0, 4.0]);
        halfspace7.fact(2.0);
        assert_eq!(halfspace7.is_equal(&halfspace8, 0.1), true);


    }
}
