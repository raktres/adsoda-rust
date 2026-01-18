//! un solide est composé de faces

use serde::{Deserialize, Serialize};
use serde_json;

use super::bases::{
    vector_normalize, vector_project, vector_subtract, Axis, Halfspace, JsonFormat, Point,
    VERY_SMALL_NUM,
};

use std::collections::HashSet;

#[derive(PartialEq, Debug)]
pub enum Orientation {
    Back,
    Front,
    Profile,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct JsonFace {
    pub face: Vec<f64>,
}

/// # définitin d'une face
///
/// # Présentation
/// create a face * This is the interface to the [`adsoda::Halfspace`] class.  
/// An Halfspace is half of an n-space.  it is described by the equation of the bounding hyperplane.  
/// A point is considered to be inside the halfspace if the left side of the equation is greater
/// than the right side when the coordinates of the point (vector) are plugged in.  
/// The first n coefficients can also be viewed as the normal vector, and the last as a
/// constant which determines which of the possible parallel hyperplanes in n-space this one is.
///
/// A halfspace is represented by the equation of the bounding hyperplane, so a Hyperplane  is really the same as a Halfspace.

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)] // Copy, Eq,
pub struct Face {
    // equ est l'équation de l'hyperplan qui contient la face.
    // peut-être faut-il le renommer en halfspace, mais problème de compatibilité avec les exports précédents
    pub halfspace: Halfspace,
    pub dim: usize,
    // id: usize,
    pub name: String,
    // color: String,
    // selected: bool,

    // equ contient l'équation de l'hyperplan,
    // mais c'est touching_corners qui permet de définir la forme de la face
    // en listant les différents sommets
    // c'est une option, quand la face n'est pas calculée, touching_corners est à None
    // #[serde(skip_serializing_if = "Option::is_none")]
    // pub touching_corners: Option<Vec<Point>>, // Vec<Point>, // TODO liste de points
    pub touching_corners: Vec<Point>, // Vec<Point>, // TODO liste de points
    // adjacent_refs: Vec<& 'a Face<'a>>,
    // il s'agit du pointeur vers les faces qui sont adjacentes
    // difficulté, comment constituer ce lien vers une liste de faces
    // plusieurs options envisageables, à définir
    // c'est une option. quand ce n'est pas calculé, c'est à None
    // #[serde(skip_serializing_if = "Option::is_none")]
    pub adjacent_refs: HashSet<usize>, // HashSet<usize>,
                                       // adjacent_refs: Option<HashSet<usize>>, // HashSet<usize>,
                                       // adjacent_refs: HashSet<usize>,    // TODO liste de faces
}

impl Face {
    /// nouvelle face
    /// quand on crée une face, on utilise juste d'équation
    pub fn new_from_halfspace(vec: &Vec<f64>) -> Self {
        // comprendre pourquoi il y avait besoin de cette transcription
        // let equ = vector_normalize(&vec.iter().map(|&x| x as f64).collect());
        let halfspace: Halfspace = Halfspace(vector_normalize(vec));
        let dim = halfspace.dimension();
        let name = "face name".to_string();
        // let color = "000".to_string();
        Face {
            halfspace,
            // id: 0,
            dim,
            name,
            // color,
            // selected: false,
            touching_corners: Vec::<Point>::new(), //Vec::<i32>, None, //
            adjacent_refs: HashSet::new(),         //None,    //
        }
    }
    pub fn new_axeface(dimension: usize, axis: &Axis) -> Self {
        let mut hype = Halfspace(Vec::<f64>::new());

        // c'est une face avec un vecteur très simple et passant par 0
        let nbhyp = dimension + 1;
        for _index in 0..nbhyp {
            hype.0.push(0.0);
        }
        hype.0[axis.0] = 1.0;

        <Face>::new_from_halfspace(&hype.to_vec())

    }

    /// sérialise dans un objet json
    ///
    /// Attention, par rapport à l'export JS, ne le met peut-être pas dans un objet face
    ///
    /// quand on exporte une face, il n'est pas nécessaire d'exporter les touching_corners ,t les adjacent_refs
    ///
    /// ## Version JS
    /// ```js
    /// function exportToJSON () {  
    ///     return `{ "face" : ${JSON.stringify(this.equ)} }`
    /// }
    /// ```
    ///
    pub fn export_to_json(&self, format: JsonFormat) -> String {
        let res: String;
        match format {
            JsonFormat::Export { .. } => {
                res = r#"{ "face" : "#.to_string()
                    + &serde_json::to_string(&self.halfspace).unwrap_or("".to_string())
                    + " }";
            }
            JsonFormat::Pretty { .. } => {
                res = "{ \"face\" : ".to_string()
                    + &serde_json::to_string_pretty(self).unwrap_or("".to_string())
                    + " \n }";
            }
            JsonFormat::Condensed { .. } => {
                res = "{ \"face\" : ".to_string()
                    + &serde_json::to_string(self).unwrap_or("".to_string())
                    + " }";
            }
        }
        res
    }

    ///
    pub fn display(&self) -> String {
        let res: String;
        res =
            "face ".to_string() + &serde_json::to_string(&self.halfspace).unwrap_or("".to_string());
        res
    }

    pub fn to_jsonface(&self) -> JsonFace {
        JsonFace {
            face: self.halfspace.to_vec(),
        }
    }

    /// nombre de faces adjacentes
    pub fn nb_adjacent_faces(&self) -> usize {
        self.adjacent_refs.len() //map_or(0, |s| s.len())
                                 /* match &self.adjacent_refs {  Some(s) => s.len(), None => 0, } */
    }

    /// Est-ce qu'il faut vraiment utiliser une option ?
    /// ne peut-on pas directement recréer un hash vide ?

    pub fn force_refresh(&mut self) {
        self.adjacent_refs.clear(); // = None;
        self.touching_corners = Vec::<Point>::new(); // None;
    }

    /// Retourne une face
    ///
    /// This method negates all terms in the equation of this halfspace.  
    ///
    /// This flips the normal without changing the boundary halfplane.
    pub fn flip(&mut self) {
        self.halfspace.flip();
        // Est-ce nécessaire ?
        // self.force_refresh();
    }

    /// translate une face selon un vecteur
    ///
    /// ne touche pas le vecteur directeur de la face
    ///
    /// on translate juste la face => on modifie la constante
    ///
    /// translate the face following the given vector.
    ///
    /// Translation doesn't change normal vector, Just the constant term need to be changed.
    ///
    ///  > new constant = old constant - dot(normal, vector)
    ///
    /// # Pres
    /// Given a halfspace
    ///
    ///  >  a1*x1 + ... + an*xn + k = 0
    ///
    ///  We can translate by vector (v1, v2, ..., vn) by substituting (xi - vi) for
    ///  all xi, yielding
    ///
    ///  >  a1*(x1-v1) + ... + an*(xn-vn) + k = 0
    ///
    ///  This simplifies to
    ///
    ///  > a1*x1 + ... + an*xn + (k - a1*v1 - ... - an*vn) = 0
    ///
    ///  So all we have to do is change the constant term.  This is as expected,
    ///  since translating should not change the normal vector (the first n-1 terms).
    ///
    pub fn translate(&mut self, vector: &Vec<f64>) {
        self.halfspace.translate(vector);
        // Est-ce nécessaire ?
        //self.force_refresh();
    }

    /// This method applies a matrix transformation to this Halfspace.
    pub fn transform(&mut self, matrix: &Vec<Vec<f64>>) {
        self.halfspace.transform(matrix);
        // Est-ce nécessaire ?
        //self.force_refresh();
    }

    /// test si une face est visible ou non
    ///
    /// Pour l'instant c'est très basique on suppose une orientation de l'espace
    /// par rapport au repère orthonormé.
    ///
    /// Il faudrait faire le test par rapport à un point qui servirait de point de vue
    fn is_back(&self, axis: &Axis) -> bool {
        self.orientation(axis) == Orientation::Back
    }

    /// projection
    ///
    pub fn projected_halfspace(&self, axis: &Axis) -> Vec<f64> {
        vector_project(&self.halfspace_to_vec(), axis)
    }

    /// il faudra que l'on remplace axe par un vecteur directeur
    /// back signifie que c'est plutot une face arrière quand on la regarde de l'axe
    /// c'est a dire que -infini est dans le demi plan
    fn orientation(&self, axis: &Axis) -> Orientation {
        let val = self.halfspace.get_coordinate(axis);
        if val < -VERY_SMALL_NUM {
            return Orientation::Back;
        }
        if val > VERY_SMALL_NUM {
            return Orientation::Front;
        }
        Orientation::Profile
    }

    // TODO: vérifier si c'est bien utilisé
    fn _is_valid_for_order(&self, point: &Point, axis: &Axis) -> bool {
        self.is_point_inside_face(point) && self.orientation(axis) != Orientation::Profile
    }

    pub fn halfspace_to_vec(&self) -> Vec<f64> {
        self.halfspace.to_vec()
    }

    /// teste
    ///  This method returns true if point  is inside the Halfspace or on the boundary.
    /// inclue la frontière
    //
    //  The point is on the inside side or the boundary of the halfspace if
    //
    //  a x  + a x  + ... + a x + k  <=  0
    //   1 1    2 2          n n
    //
    //  where all ai are the same as in the equation of the hyperplane.
    //  The following code evaluates the left side of this inequality.
    //
    pub fn is_point_inside_or_on_face(&self, point: &Point) -> bool {
        self.halfspace.position_point(&point) > -VERY_SMALL_NUM
    }

    /// teste
    ///     //
    ///  The point is on the inside side of the halfspace if
    ///
    ///  a x  + a x  + ... + a x + k  <  0
    ///   1 1    2 2          n n
    ///
    ///  where all ai are the same as in the equation of the hyperplane.
    ///  The following code evaluates the left side of this inequality.
    ///
    pub fn is_point_inside_face(&self, point: &Point) -> bool {
        // dbg!((&self.equ, &point.position));
        self.halfspace.position_point(&point) > VERY_SMALL_NUM
    }

    /// Teste
    pub fn is_point_on_face(&self, point: &Point) -> bool {
        let pos = self.halfspace.position_point(&point);
        pos > -VERY_SMALL_NUM && pos < VERY_SMALL_NUM
    }

    // ne vérifie pas que l'axe est dans la limite
    fn pv_factor(&self, axis: &Axis) -> f64 {
        self.halfspace.get_coordinate(axis)
    }

    /// après les calculs sur le solide,
    /// une face contient un certain nombre de points
    /// elle ne peut exister réellement que si ce nombre de point supérieur à la dimension de la face
    pub fn is_real_face(&self) -> bool {
        &self.touching_corners.len() >= &self.dim
        /*
        match &self.touching_corners {
            Some(s) => s.len() >= self.dim,
            None => false,
        }
        */
    }

    /// ajoute un point à la liste des coins
    ///
    /// on vérifie que le point n'est pas déjà présent
    ///
    /// ESt-ce vraiment utile d'utiliser une option ???

    pub fn suffix_touching_corners(&mut self, corner: &Point) {
        // pour ajouter un point, il faut déjà que l'on transforme l'option de none en some
        /*
        if let None = self.touching_corners {
            self.touching_corners = Some(Vec::<Point>::new());
        }
        */
        // si touching_corners est bien un some
        // if let Some(ref mut v) = self.touching_corners {
        // on vérifie pour chacun des points de touching corner
        // que le candidat n'est pas déjà dans la liste
        // if v.iter().any(|corn| corn.is_equal(&corner)) {          v.push(corner.clone());         }
        // }
        if !self
            .touching_corners
            .iter()
            .any(|corn| corn.is_equal(&corner))
        {
            self.touching_corners.push(corner.clone());
        }
    }

    /// concerne les projections
    /// un peu plus compliqué que le cas des coupes pour gérer les faces masquées
    fn intersection_into_silhouette_face(&self, adja_face: &Face, axis: &Axis) -> Option<Face> {
        // le coef de la face adjacente
        let adjacent_factor = adja_face.pv_factor(axis);
        let self_factor = self.pv_factor(axis);

        let adjacent_eq: Vec<f64> = adja_face
            .halfspace
            .to_vec()
            .iter()
            .map(|&x| x * self_factor)
            .collect();
        let self_eq: Vec<f64> = self
            .halfspace
            .to_vec()
            .iter()
            .map(|&x| x * adjacent_factor)
            .collect();

        // comme ça on est sur que le coefficient selon l'axe est bien nul.
        // la face de silhouette est bien orthogonale à cet axe
        let diff_eq = vector_subtract(&self_eq, &adjacent_eq);

        // if let None = adja_face.touching_corners {            return None;        }
        if adja_face.touching_corners.len() == 0 {
            return None;
        }

        // let a_tc = adja_face.touching_corners.unwrap_or(Vec::<Point>::new());
        // if let Some(ref a_tc) = adja_face.touching_corners {
        let out_point = adja_face
            .touching_corners
            .iter()
            .find(|point| !self.is_point_on_face(point));

        let out_point = match out_point {
            Some(point) => point.clone(),
            None => return None,
        };
        // looking for a point in solid, but not on main face
        // for exemple, a touching corner of the adjacent face
        // not common with main face

        let mut n_face = Face::new_from_halfspace(&diff_eq);

        // flip the face if point is not inside
        if !n_face.is_point_inside_face(&out_point) {
            n_face.flip();
        }

        n_face.name = format!("proj de {:?} selon {:?}", self.halfspace, axis);

        Some(n_face)
    }

    /// TODO: probablement moyen de factoriser avec l'autre
    /// concerne le coupes
    /// relativement simple car la face que l'on considère est le plan de coupe,
    /// tout est donc visible
    fn intersection_cut_into_silhouette_face(&self, adja_face: &Face, axis: &Axis) -> Face {
        let adjacent_factor = adja_face.pv_factor(axis);
        let self_factor = self.pv_factor(axis);

        let adjacent_eq: Vec<f64> = adja_face
            .halfspace
            .to_vec()
            .iter()
            .map(|&x| x * self_factor)
            .collect();
        let self_eq: Vec<f64> = self
            .halfspace
            .to_vec()
            .iter()
            .map(|&x| x * adjacent_factor)
            .collect();

        // comme ça on est sur que le coefficient selon l'axe est bien nul.
        // la face de silhouette est bien orthogonale à cet axe
        let diff_eq = vector_subtract(&adjacent_eq, &self_eq);

        // la normalisation du vecteur est dans new_from_halfspace
        let mut new_face = Face::new_from_halfspace(&diff_eq);
        new_face.name = format!("cut de {:?} pour {:?} = 0", self.halfspace, axis);

        new_face
    }

    pub fn silhouette(&self, axis: &Axis, faces: &Vec<Face>) -> Option<Vec<Face>> {
        /*
        if let None = self.adjacent_refs {
            return None;
        }
        */
        // si la face n'est pas orienté vers l'axis, forcément on ne peut pas la voir
        if self.is_back(axis) {
            return None;
        }

        // s'il y a bien une liste des références des faces adjacentes
        // if let Some(ref adja_refs) = self.adjacent_refs {
        // on initialise une liste de faces qui vont être la silhouette
        let mut sil_faces = Vec::<Face>::new();

        // balaie les références des faces adjacentes
        // TODO: est-ce que que l'on est vraiment obligé d'utiliser clone ???
        for adja_ref in self.adjacent_refs.clone().drain() {
            // on ne s'intéresse qu'aux faces adjacente qui sont
            // tournées vers l'arrière.
            // en effet si la face adjacente a la même orientation
            // que la face, ça ne matérialise pas le contour d'une
            // silhouette
            if faces[adja_ref].is_back(axis) {
                // si l'intersection de la face avec la face adjacente existe, on l'ajoute
                // à la liste des faces de silhouettes
                if let Some(nface) = self.intersection_into_silhouette_face(&faces[adja_ref], axis)
                {
                    sil_faces.push(nface);
                }
            }
        }

        if sil_faces.len() > 0 {
            return Some(sil_faces);
        } else {
            return None;
        }
        //} else {             return None;         }
    }

    /// la sihouette est une vue en projection
    /// alors que cut_silhouette est la vue en coupe.
    pub fn cut_silhouette(&self, axis: &Axis, faces: &Vec<Face>) -> Option<Vec<Face>> {
        //if let None = self.adjacent_refs {             return None;         }
        // if let Some(ref adja_refs) = self.adjacent_refs {
        // dbg!(&self);
        println!(
            "axe {:?} nbre de faces {} et self {:?}",
            axis,
            faces.len(),
            self
        );
        let mut sil_faces = Vec::<Face>::new();

        // TODO! vraiment obligé d'utiliser clone ???
        for adja_ref in self.adjacent_refs.clone().drain() {
            // contrairement à une silhouette normale, on ne prend que la silhouette de la face
            let nface = self.intersection_cut_into_silhouette_face(&faces[adja_ref], axis);
            sil_faces.push(nface);
        }

        return Some(sil_faces);
        // } else {             return None;        }
    }

    /// 3D specific
    /// il n'est possible d'ordonner les coins que si ce n'est pas un none.
    // TODO: retourner None ?
    // TODO: reprendre, voir s'il y a vraiment besoin de cloner
    pub fn d3_ordered_corners(&self) -> Vec<Point> {
        // if let None = self.touching_corners {        return Vec::<Point>::new();        }
        // si touching_corners est bien un some
        // if let Some(ref v) = self.touching_corners {
        // on vérifie pour chacun des points de touching corner
        // que le candidat n'est pas déjà dans la liste
        let corners = self.touching_corners.clone();
        if corners.len() < 2 {
            return corners; // Pas assez de coins pour trier
        }

        let ci = [corners[0].0[0], corners[0].0[1], corners[0].0[2]];
        let vequ = [
            self.halfspace.0[0],
            self.halfspace.0[1],
            self.halfspace.0[2],
        ];
        let vref = [
            ci[0] - corners[1].0[0],
            ci[1] - corners[1].0[1],
            ci[2] - corners[1].0[2],
        ];

        let mut ordered: Vec<(f64, Point)> = corners
            .into_iter()
            .map(|corner| {
                (
                    d3_order(&[corner.0[0], corner.0[1], corner.0[2]], &vequ, &ci, &vref),
                    corner,
                )
            })
            .collect();
        // supprimer unwrap
        ordered.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        return ordered.into_iter().map(|el| el.1).collect();
        // } else {            return Vec::<Point>::new();        }
    }

    pub fn fact_size (&mut self,factor: f64) {
        self.halfspace.fact(factor);
  }

}

impl Point {
    // fonction statique, à regrouper ?
    pub fn is_inside_faces(&self, faces: &Vec<Face>) -> bool {
        faces.iter().all(|face| face.is_point_inside_face(self))
    }
}

/* // est-ce que le clone est obligatoire ?
// est-ce toujours utile ?
pub fn faces_intersection(faces: &Vec<Face>) -> Option<Vec<f64>> {
    let hyps: Vec<Vec<f64>> = faces.iter().map(|face| face.halfspace.to_vec()).collect();
    hyperplanes_intersect(&hyps)
} */

// est-ce que le clone est obligatoire ?

/// spécifique 3D
/// expliquer fonctionnement
/// TODO: utiliser les objets points
fn d3_order(
    point1: &[f64; 3],
    halfspace: &[f64; 3],
    pointref: &[f64; 3],
    vectorref: &[f64; 3],
) -> f64 {
    let v1 = [
        point1[0] - pointref[0],
        point1[1] - pointref[1],
        point1[2] - pointref[2],
    ];

    let cross_p = [
        vectorref[1] * v1[2] - vectorref[2] * v1[1],
        vectorref[2] * v1[0] - vectorref[0] * v1[2],
        vectorref[0] * v1[1] - vectorref[1] * v1[0],
    ];

    let norm = (cross_p[0] * cross_p[0] + cross_p[1] * cross_p[1] + cross_p[2] * cross_p[2]).sqrt();
    let dot_p = vectorref[0] * v1[0] + vectorref[1] * v1[1] + vectorref[2] * v1[2];
    let theta = dot_p.atan2(norm);
    let sign = cross_p[0] * halfspace[0] + cross_p[1] * halfspace[1] + cross_p[2] * halfspace[2];

    if sign < 0.0 {
        -theta
    } else {
        theta
    }
}

/*

let face1 = Face::new(vec![1.0, 2.0, 3.0]);
    let face2 = Face::new(vec![4.0, 5.0, 6.0]);
    let axe = Axe;

    if let Some(new_face) = face1.intersection_into_silhouette_face(&face2, &axe) {
        println!("New face name: {}", new_face.name);
    } else {
        println!("No valid intersection found.");
    }


        let point1 = [1.0, 2.0, 3.0];
    let halfspace = [0.0, 0.0, 1.0];
    let pointref = [0.0, 0.0, 0.0];
    let vectorref = [1.0, 0.0, 0.0];

    let result = order_3d(&point1, &halfspace, &pointref, &vectorref);
    println!("Order 3D result: {}", result);
*/

#[cfg(test)]
mod tests_face {
    use super::*;
    // use approx::assert_abs_diff_eq;
    #[test]
    pub fn face_validation() {
        let eq = vec![1.0, 2.0, 3.0];
        let f1 = Face::new_from_halfspace(&eq);
        println!("face {:?}", f1);

        println!(
            "face to json condensed {}",
            f1.export_to_json(JsonFormat::Condensed)
        );
        println!(
            "face to json prety {}",
            f1.export_to_json(JsonFormat::Pretty)
        );
        dbg!(f1.export_to_json(JsonFormat::Export));

        let mut face1 = Face::new_from_halfspace(&vec![1.0, 0.0, 1.0]);
        face1.flip();
        // let v2 = vec![-1.0, 1.0, 3.0];

        assert_eq!(face1.halfspace.to_vec(), vec![-1.0, 0.0, -1.0]);
        face1.flip();
        face1.translate(&vec![1.0, 1.0]);

        assert_eq!(face1.halfspace.to_vec(), vec![1.0, 0.0, 0.0]);

        let mut face2 = Face::new_from_halfspace(&vec![1.0, 1.0, 0.0]);
        // equ est normalisé en [0.7071067811865475, 0.7071067811865475, 0.7071067811865475]
        face2.transform(&vec![vec![2.0, 0.0], vec![0.0, 2.0]]);
        assert_eq!(
            face2.halfspace.to_vec(),
            vec![0.7071067811865476, 0.7071067811865476, -0.0]
        );

        let mut face3 = Face::new_from_halfspace(&vec![1.0, 0.0, 10.0]);
        // equ est normalisé en [0.7071067811865475, 0.7071067811865475, 0.7071067811865475]
        face3.transform(&vec![vec![0.0, 1.0], vec![-1.0, 0.0]]);
        assert_eq!(face3.halfspace.to_vec(), vec![0.0, -1.0, 10.0]);

        let mut face4 = Face::new_from_halfspace(&vec![1.0, 1.0, 10.0]);
        // equ est normalisé en [0.7071067811865475, 0.7071067811865475, 0.7071067811865475]
        face4.transform(&vec![vec![0.0, 2.0], vec![-1.0, 0.0]]);
        assert_eq!(
            face4.halfspace.to_vec(),
            vec![0.8944271909999159, -0.4472135954999579, 4.47213595499958]
        );

        assert_eq!(face4.orientation(&Axis(1)), Orientation::Back);
        assert_eq!(face4.orientation(&Axis(0)), Orientation::Front);
        assert_eq!(face3.orientation(&Axis(0)), Orientation::Profile);
        assert_eq!(face4.is_back(&Axis(1)), true);
        assert_eq!(face4.is_back(&Axis(0)), false);

        let face5 = Face::new_from_halfspace(&vec![1.0, -1.0, 3.0]);
        let p0 = Point(vec![0.0, 0.0]);
        let p1 = Point(vec![6.0, 1.0]);
        assert_eq!(face5.is_point_inside_face(&p1), true);
        let p2 = Point(vec![6.0, 9.0]);
        let p3 = Point(vec![0.0, 10.0]);
        assert_eq!(face5.is_point_inside_face(&p2), false);
        assert_eq!(face5.is_point_inside_face(&p3), false);
        assert_eq!(face5.is_point_inside_or_on_face(&p1), true);
        assert_eq!(face5.is_point_inside_or_on_face(&p2), true);
        assert_eq!(face5.is_point_inside_or_on_face(&p3), false);
        assert_eq!(face5.is_point_on_face(&p1), false);
        assert_eq!(face5.is_point_on_face(&p2), true);
        assert_eq!(face5.is_point_on_face(&p3), false);

        let face6 = Face::new_from_halfspace(&vec![1.0, 1.0, -5.0]);
        assert_eq!(face6.is_point_inside_face(&p1), true);
        assert_eq!(face6.is_point_inside_face(&p0), false);

        assert_eq!(
            p1.is_inside_faces(&vec![face5.clone(), face6.clone()]),
            true
        );
        assert_eq!(p0.is_inside_faces(&vec![face5, face6]), false);
    }
}

