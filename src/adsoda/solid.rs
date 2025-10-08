//! Solid
//!
//! Fast
//! [`Easy`]
//!
//! [`Easy`]: bolpo époé
///
use serde::{Deserialize, Serialize};

use super::bases::{
    among_index, hyperplanes_intersect, vector_flip, Axis, JsonFormat, Point, ProjecType,
    MAX_FACE_PER_CORNER, VERY_SMALL_NUM,
};
use super::face::{Face, JsonFace};

// use super::{bases, face};

// use std::collections::HashMap;
use uuid::Uuid;
// use std::collections::HashMap;

pub trait NdSolid {
    fn translate(&mut self, vector: &Vec<f64>);
    fn transform(&mut self, transformation: &Vec<Vec<f64>>, center: &Point);
    fn compute(&mut self, mode: ComputeMode);
    fn define_center(&mut self);
}

/// paramètre interne pour factorisation des calculs
#[derive(Default)]
pub enum ComputeMode {
    #[default]
    /// le calcul du solide intègre la définition des projections
    Full,
    /// le calcul du solide ne fait le calcul des projections.
    /// Essentiellement utilisé pour les coupes.
    Basic,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct JsonSolid {
    solidname: String,
    dimension: usize,
    color: String,
    faces: Vec<JsonFace>,
}




/// # Solid
/// faces, dimension..
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)] // Copy, Eq,
pub struct Solid {
    pub dimension: usize,
    pub faces: Vec<Face>,
    pub corners: Vec<Point>,
    pub silhouettes: Vec<Vec<Face>>,
    pub cutsilhouettes: Vec<Vec<Face>>,
    pub center: Vec<f64>,
    pub name: String,
    pub color: String,
    ready: bool,
    pub uuid: Uuid,
}

impl Solid {
    pub fn new() -> Self {
        let dimension = 0;
        let name = "solid name".to_string();
        let color = "000".to_string();
        // let center = None; // Point::new(vec![0.0,0.0,0.0]);

        Solid {
            dimension,
            name,
            color,
            ready: false,
            faces: Vec::<Face>::new(),
            corners: Vec::<Point>::new(),
            silhouettes: Vec::<Vec<Face>>::new(),
            cutsilhouettes: Vec::<Vec<Face>>::new(),
            center: Vec::<f64>::new(),
            uuid: Uuid::new_v4(),
        }
    }

    fn new_from_jsonsolid(sol: JsonSolid) -> Self {
        let dimension = sol.dimension;
        let name = sol.solidname;
        let color = sol.color;
        let faces: Vec<Face> = sol
            .faces
            .iter()
            .map(|jsf| <Face>::new_from_halfspace(&jsf.face))
            .collect();
        /*
                [...json.faces].forEach(fac => {
          sol.addFace(Face.importFromJSON(fac))
        })
          */

        Solid {
            dimension,
            name,
            color,
            ready: false,
            faces,
            corners: Vec::<Point>::new(),
            silhouettes: Vec::<Vec<Face>>::new(),
            //silhouettes: Vec::<Solid>::new(),
            cutsilhouettes: Vec::<Vec<Face>>::new(),
            center: Vec::<f64>::new(),
            uuid: Uuid::new_v4(),
        }
    }
    /// create a face from a json
    /// du json ou un string en entrée ?
    ///
    ///
    pub fn new_from_json(json: String) -> Self {
        // let parsed_solid =
        match serde_json::from_str(&json) {
            Ok(parsed_solid) => Solid::new_from_jsonsolid(parsed_solid),
            Err(_e) => Solid::new(),
        }
    }

    pub fn export_to_json(&self, format: JsonFormat) -> String {
        let res: String;
        match format {
            JsonFormat::Export { .. } => {
                let jss = JsonSolid {
                    solidname: self.name.clone(),
                    dimension: self.dimension,
                    color: self.color.clone(),
                    faces: self.faces.iter().map(|f| f.to_jsonface()).collect(),
                };

                res = serde_json::to_string(&jss).unwrap_or("".to_string());
            }
            JsonFormat::Pretty { .. } => {
                res = "{ \"solid\" : ".to_string()
                    + &serde_json::to_string_pretty(self).unwrap_or("".to_string())
                    + " \n }";
            }
            JsonFormat::Condensed { .. } => {
                res = "{ \"solid\" : ".to_string()
                    + &serde_json::to_string(self).unwrap_or("".to_string())
                    + " }";
            }
        }
        res
    }

    ///
    fn suffix_face(&mut self, face: &Face) {
        self.faces.push(face.clone());
    }

    ///
    fn suffix_corner(&mut self, corner: Point) -> bool {
        // if let None = self.corners {            self.corners = Some(Vec::<Point>::new());        }
        //if let Some(ref mut corners) = self.corners {
        if !self
            .corners
            .iter()
            //.any(|corn| is_corner_equal(corn, &corner, VERY_SMALL_NUM)) {
            .any(|corn| corn.is_equal(&corner))
        {
            self.corners.push(corner);
            return true;
        } else {
            return false;
        }
    }

    /// # Force le rafraichissement d'un solide
    fn force_refresh(&mut self) {
        self.ready = false;
        // refresh des faces
        for face in &mut self.faces {
            face.force_refresh();
        }
        self.corners = Vec::<Point>::new();
        self.center = Vec::<f64>::new();
        // utile de le séparer pour quand on pourra bouger le vecteur d'observation
        self.silhouettes = Vec::<Vec<Face>>::new();
        self.cutsilhouettes = Vec::<Vec<Face>>::new();
    }
    ///
    /// TODO: voir si on peut reprendre pour éviter le clone !!!
    fn filter_real_faces(&mut self) {
        // Créez un nouveau vecteur pour stocker les faces filtrées
        // let filtered_faces: Vec<Face>
        self.faces = self
            .faces
            .iter()
            .filter(|face| {
                (face.is_real_face())
                    && (face
                        .touching_corners
                        .iter()
                        .any(|corner| self.is_point_inside_or_on(corner)))
            })
            .cloned()
            .collect();
    }

    /* fn is_point_inside(&self, point: &Point) -> bool {
        self.faces
            .iter()
            .all(|face| face.is_point_inside_face(point))
    } */

    ///
    fn is_point_inside_or_on(&self, point: &Point) -> bool {
        self.faces
            .iter()
            .all(|face| face.is_point_inside_or_on_face(point))
    }

    ///
    fn add_face(&mut self, face: &Face) {
        self.suffix_face(face);
        self.force_refresh();
    }

    /// mutation de self et retour d'un nouveau solide
    pub fn slice_with(&mut self, face: Face) -> Solid {
        let mut flipped_face: Face = face.clone();
        flipped_face.flip();

        // let face = Face::new_from_halfspace(vec)
        let mut solid1 = self.clone();
        solid1.name = self.name.clone() + "/outer/";
        self.name = self.name.clone() + "/inner/";

        solid1.add_face(&flipped_face);
        self.add_face(&face);
        solid1
    }

    ///
    pub fn is_non_empty_solid(&self) -> bool {
        // TODO: juste retrouver un point sur une des faces
        true
        /*
        match &self.corners {
            Some(s) => self.dimension < s.len(),
            None => false,
        }
        */
    }

    ///
    fn is_corner_added(&mut self, corner: Point) -> bool {
        if !self.is_point_inside_or_on(&corner) {
            // dbg!("pas dans le solide");
            return false;
        }
        self.suffix_corner(corner)
    }

    ///
    fn process_corner(&mut self, facesref: &Vec<usize>, corner: &Point) {
        // dbg!(&corner);

        if self.is_corner_added(corner.clone()) {
            // dbg!("point bien ajouté !");
            //  update_adjacent_faces_refs(&mut self.faces, facesref, corner)

            // update adjacent faces refs
            // fn update_adjacent_faces_refs(faces: &mut Vec<Face>, refs: &Vec<usize>, corner: &Point) {
            for &ref_index in facesref {
                self.faces[ref_index].suffix_touching_corners(corner);
                // if let None = faces[ref_index].adjacent_refs {             faces[ref_index].adjacent_refs = Some(HashSet::<usize>::new());        }
            }

            let grouprefs = among_index(facesref.len(), 2, 2);
            for groupref in grouprefs {
                // faces[refs[groupref[0]]].adjacent_refs.as_mut().map(|a| a.insert(refs[groupref[1]]));
                self.faces[facesref[groupref[0]]]
                    .adjacent_refs
                    .insert(facesref[groupref[1]]);
                self.faces[facesref[groupref[1]]]
                    .adjacent_refs
                    .insert(facesref[groupref[0]]);
                /*
                if let Some(ref mut a) = faces[refs[groupref[0]]].adjacent_refs {
                    a.insert(refs[groupref[1]]);
                }
                if let Some(ref mut a) = faces[refs[groupref[1]]].adjacent_refs {
                    a.insert(refs[groupref[0]]);
                }
                */
            }
        } else {
            //
        }
    }

    ///
    fn compute_intersection(&mut self, facesref: &Vec<usize>) {
        // recherche d'une intersection

        // est-ce que le clone est obligatoire ?

        let hyps: Vec<Vec<f64>> = facesref
            .iter()
            .map(|&ref_index| self.faces[ref_index].halfspace.to_vec())
            .collect();

        let intersection = hyperplanes_intersect(&hyps);
        // dbg!(&intersection);
        // TODO: pourquoi l'intersection pourrait ne pas être sur les faces ? mais en fait si !!!

        match intersection {
            // && facesref.every(ref => this.faces[ref].isPointOnFace(intersection))
            Some(intersect) => self.process_corner(facesref, &Point(intersect)), // TODO modifier facse_ref_inters corner pour gérer un Point
            None => (),
        }
    }

    /// prend toutes les couples de faces et calcule les intersections
    ///
    ///
    pub fn compute_adjacencies(&mut self) {
        // dbg!(&self);
        let groups_faces = among_index(self.faces.len(), self.dimension, MAX_FACE_PER_CORNER); // this.dimension) //
                                                                                               // dbg!(&groups_faces.len());
                                                                                               // dbg!(&groups_faces);
        let n = groups_faces.len();
        for index in 0..n {
            // dbg!(index);
            self.compute_intersection(&groups_faces[index])
        }
    }

    /// une fois que le solid est constitué de face
    /// on utilise l'ensemble des faces pour trouver leurs intersections
    /// il faut d'abord avoir calculé les faces adjacentes
    pub fn compute_corners(&mut self) {
        // pour définir un point il faut un intersection d'hyperplans
        let grp_dimension = self.dimension - 1;
        // on liste l'ensemble des équations des facse constitant le solide
        // nécessaire pour gérer des droits sur la mutabilité de self
        let faces_equations: Vec<Vec<f64>> = self
            .faces
            .iter()
            .map(|face| face.halfspace_to_vec())
            .collect();

        // on itere sur chacune des faces pour peupler ses suffix.touching corners
        // mais pourquoi on ne suffix pas aussi les faces adjacentes ???
        // en fait on fait 2* trop de calculs !!!
        // plutot que de travailler face par face, il faut travailler sur les groupes
        for face in self.faces.iter_mut() {
            let nb_adjacent_faces = face.nb_adjacent_faces();
            let groups_ref_faces = among_index(nb_adjacent_faces, grp_dimension, grp_dimension);
            let mut adjacents_equ = Vec::<Vec<f64>>::new();
            for adjaref in face.adjacent_refs.clone().drain() {
                adjacents_equ.push(faces_equations[adjaref].clone());
            }

            for group in groups_ref_faces {
                let eqs: Vec<Vec<f64>> =
                    group.iter().map(|&id| adjacents_equ[id].clone()).collect();

                let intersection = hyperplanes_intersect(&eqs);
                if let Some(ref inter) = intersection {
                    let point = Point(inter.clone());
                    // une intersection peut ne pas être dans le solide, il faut vrifier
                    // on ne peut pas utiliser self.is_point_inside_or_on_solid car self déjà modifié !!!
                    if point.is_inside_or_on_halfspaces(&faces_equations) {
                        face.suffix_touching_corners(&point);
                    }
                }
            }
        }
        // est-ce nécessaire d'entretenir la liste des points ?
        /*
        let cornerList = []
        // TODO, voir pour remplacer uniq par la fonction infra
        this.faces.forEach(face => {
          cornerList = cornerList.concat([...face.touchingCorners])
        })
        _t.corners.length = 0
        cornerList.forEach((corner, idx) => {
          for (let index = idx + 1; index < cornerList.length; index++) {
            if (isCornerEqual(corner, cornerList[index])) { return false }
          }
          _t.corners.push(corner)
        })

            */
    }

    ///

    ///
    fn create_silhouette(&self, axis: &Axis, project_type: ProjecType) -> Vec<Face> {
        // let sFaces = self.faces;
        // une sihouette est un ensemble de facs
        let mut sil = Vec::<Face>::new();

        // n le nombre de faces
        let n = self.faces.len();

        match project_type {
            ProjecType::Cut => {
                // dbg!(" parti pour un cut !");

                // Il faut crééer le plan de coupe
                let mut cut_face: Face = Face::new_axeface(self.dimension, axis);
                let cut_halfspace = cut_face.halfspace.clone();
                // c'est une face avec un vecteur très simple et passant par 0

                // on va créer un nouveau solid qui sera coupé par le plan.
                // on ne touche pas au solid initial
                let mut solid1 = self.clone();
                solid1.name = self.name.clone() + "/ cut /" + &axis.to_string();

                // attention ! le plan de coupe peut déjà être une face du solide

                if let Some(findedface) = solid1
                    .faces
                    .iter()
                    .find(|face| cut_halfspace.is_equal(&face.halfspace, 0.0))
                {
                    cut_face = findedface.clone();
                } else {
                    // si le plan de coupe n'est pas déjà une face, on coupe solid1
                    // cut_face = <Face>::new_from_halfspace(&hype.to_vec());
                    cut_face.name = "cut".to_string();

                    solid1.add_face(&cut_face);

                    solid1.compute(ComputeMode::Basic);

                    // dbg!(&solid1);
                    // dbg!(&solid1);

                    // pb, cut_face ne récupère pas les éléments de face adajcente calculés dans solid1
                    // il faut donc extraire à nouveau cette face
                    if let Some(findedface) = solid1
                        .faces
                        .iter()
                        .find(|face| cut_halfspace.is_equal(&face.halfspace, 0.0))
                    {
                        cut_face = findedface.clone();
                    }
                    // dbg!(&cut_face);
                }

                //dbg!(&cut_face);
                // TODO:  ne fonctionne pas !!!

                // let tmp_res = ;
                //  dbg!(tmp_res);
                // dbg!(&solid1.faces.len());
                // dbg!(&cut_face);

                match cut_face.cut_silhouette(axis, &solid1.faces) {
                    Some(s) => sil = s,
                    None => (),
                }
                // dbg!(&sil);

                /*
                if (!cface) {
                  cface = new Face(hype)
                  cface.name = 'cut'
                  solid1.addFace(cface)
                  solid1.ensureFaces()
                }
                const sFaces = [...solid1.faces]
                const sil = cface.cutSilhouette(axis, sFaces, this.center)
                */
            }
            ProjecType::Projection => {
                dbg!(" parti pour une projection !");
                for index in 0..n {
                    if let Some(new_sil) = self.faces[index].silhouette(axis, &self.faces) {
                        sil.extend(new_sil);
                    }
                }
            }
        }
        // dbg!(" fin de projection / cut - début de filtre");

        let nb = sil.len();
        let mut filtered_sil = Vec::<Face>::new();
        for idx in 0..nb {
            let mut res = true;
            for index in (idx + 1)..nb {
                if sil[idx]
                    .halfspace
                    .is_equal(&sil[index].halfspace, VERY_SMALL_NUM)
                {
                    res = false;
                    break;
                }
            }
            if res {
                filtered_sil.push(sil[idx].clone())
            }
        }
        // dbg!("fin proj / cut");
        return filtered_sil;
    }

    pub fn get_silhouette(&self, axis: Axis, projectype: ProjecType) -> &Vec<Face> {
        match projectype {
            ProjecType::Projection => {
                return &self.silhouettes[axis.0];
            }
            ProjecType::Cut => {
                return &self.cutsilhouettes[axis.0];
            }
        }
    }
    /*
    fn get_cutsilhouette(&self, axis: Axis) -> &Vec<Face> {
        return &self.cutsilhouettes[axis];
    }
    */

    /// si bool est à true, alors on fait une coupe
    pub fn project(&mut self, axis: &Axis, project_type: ProjecType) -> Option<Solid> {
        // attention, dans certains cas, il a fallu forcer le recalcul du solide
        // this.unvalidSolid()
        if !self.ready {
            // this.ensureFaces()
            // this.ensureSilhouettes()
            dbg!("pas pret !");
            return None;
        }

        let mut proj_solid = <Solid>::new();
        proj_solid.dimension = self.dimension - 1;
        proj_solid.color = self.color.clone();
        // let mut halfspaces = Vec::<Vec<f64>>::new();
        // let nbsil;
        match project_type {
            ProjecType::Projection => {
                dbg!("projection");
                // nbsil = self.silhouettes[axis.0].len();

                /* for i in 0..nbsil {
                    // utilisation un peu violente de projecVector pour projeter la silhouette
                    proj_solid.add_face(&Face::new_from_halfspace(&self.silhouettes[axis.0][i].projected_halfspace(axis)));
                    // halfspaces.push(self.silhouettes[axis.0][i].projected_halfspace(axis));
                    /* halfspaces.push(vector_project(
                        &self.silhouettes[axis.0][i].halfspace_to_vec(),
                        axis,
                    )); */
                } */
                for sil in &self.silhouettes[axis.0] {
                    proj_solid.add_face(&Face::new_from_halfspace(&sil.projected_halfspace(axis)));
                }
                proj_solid.name =
                    "projection ".to_owned() + &self.name + "|P" + &axis.0.to_string() + "|"
            }
            ProjecType::Cut => {
                dbg!("coupe");
                for sil in &self.cutsilhouettes[axis.0] {
                    proj_solid.add_face(&Face::new_from_halfspace(&sil.projected_halfspace(axis)));
                }
                proj_solid.name =
                    "axis cut ".to_owned() + &self.name + "|P" + &axis.0.to_string() + "|"
                /* nbsil = self.cutsilhouettes[axis.0].len();
                for i in 0..nbsil {
                    // utilisation un peu violente de projecVector pour projeter la silhouette
                    // halfspaces.push(self.cutsilhouettes[axis.0][i].projected_halfspace(axis));
                    proj_solid.add_face(&Face::new_from_halfspace(&self.cutsilhouettes[axis.0][i].projected_halfspace(axis)));
                    /* halfspaces.push(vector_project(
                        &self.cutsilhouettes[axis.0][i].halfspace_to_vec(),
                        axis,
                    )); */
                } */
            }
        }

        // TODO: Voir s'il y a besoin de filtrer les HS, utilisr iSHSEquel
        // trouver d'où peut provenir cette HS à 0. Apparait pour les cubes

        // solid.add_faces_from_halfspaces(&halfspaces);
        // TODO color of projection is function of lights

        /* match project_type {
            ProjecType::Projection => {
                proj_solid.name =
                    "projection ".to_owned() + &self.name + "|P" + &axis.0.to_string() + "|"
            }
            ProjecType::Cut => {
                proj_solid.name = "axis cut ".to_owned() + &self.name + "|P" + &axis.0.to_string() + "|"
            }
        } */
        // solid.parentUuid = self.uuid;

        proj_solid.compute(ComputeMode::Basic);

        if proj_solid.corners.len() > proj_solid.dimension {
            return Some(proj_solid);
        } else {
            return None;
        }
    }

    
}

impl NdSolid for Solid {
    /// # Translate un solide
    ///
    /// Pour cela, on translate toutes les faces et on force à recalculer le solide.
    fn translate(&mut self, vector: &Vec<f64>) {
        for face in &mut self.faces {
            face.translate(vector);
        }
        self.force_refresh();
    }

    /// # Applique une matrice de transfomation à un solide
    ///
    /// on applique la transformation à chacune des faces
    /// puis ensuite on force le recalcul du solide
    ///
    /// Pour appliquer la transformation, on commence par mettre le solide à l'origine,
    /// on lui applique la matrice, puis on le ramène à sa position initiale
    ///
    fn transform(&mut self, transformation: &Vec<Vec<f64>>, center: &Point) {
        let centerl = vector_flip(&center.to_vec());
        for face in &mut self.faces {
            face.translate(&centerl);
            face.transform(transformation);
            face.translate(&center.to_vec());
        }
        self.force_refresh();
    }
    fn compute(&mut self, mode: ComputeMode) {
        //self.erase_old_adjacencies();
        self.compute_adjacencies();
        // TODO: erreur dans certains cas

        let nba = self.faces.len();
        // println!(" attention  nba {}", nba);
        self.filter_real_faces();
        // println!(" attention  nouv nb {}", self.faces.len());
        if self.faces.len() != nba {
            // self.erase_old_adjacencies();
            self.force_refresh();
            // println!(" recalcul des adjacencies");
            self.compute_adjacencies();
            // dbg!(&self);
        }

        // on calcule les silhouettes tout de suite, sauf si on est déjà en train de calculer une silhouette !
        match mode {
            ComputeMode::Full => {
                let dim = self.dimension;
                for i in 0..dim {
                    self.silhouettes
                        .push(self.create_silhouette(&Axis(i), ProjecType::Projection));
                    self.cutsilhouettes
                        .push(self.create_silhouette(&Axis(i), ProjecType::Cut));
                }
            }
            ComputeMode::Basic => (),
        }

        self.define_center();

        self.ready = true;
    }

    fn define_center(&mut self) {
        let mut middle = Vec::<f64>::new();

        let mut positions: Vec<Vec<f64>> = Vec::<Vec<f64>>::new();
        let nb = self.corners.len();
        for i in 0..nb {
            positions.push(self.corners[i].to_vec());
        }
        for i in 0..self.dimension {
            let mut min_val = f64::MAX;
            let mut max_val = f64::MIN;
            for posidx in 0..nb {
                if positions[posidx][i] > max_val {
                    max_val = positions[posidx][i];
                }
                if positions[posidx][i] < min_val {
                    min_val = positions[posidx][i];
                }
            }
            middle.push((max_val + min_val) / 2.0);
        }
        self.center = middle;
    }
}


/* #[derive(Clone, PartialEq, Debug, Serialize, Deserialize)] // Copy, Eq,
pub struct Group {
    pub solids: HashMap<Uuid, Uuid>,
    pub center: Vec<f64>,
    pub name: String,
    pub uuid: Uuid,
    pub parent: Option<Uuid>,
    ready: bool,
} */

//
/* fn update_adjacent_faces_refs(faces: &mut Vec<Face>, refs: &Vec<usize>, corner: &Point) {
    for &ref_index in refs {
        faces[ref_index].suffix_touching_corners(corner);
        // if let None = faces[ref_index].adjacent_refs {             faces[ref_index].adjacent_refs = Some(HashSet::<usize>::new());        }
    }

    let grouprefs = among_index(refs.len(), 2, 2);
    for groupref in grouprefs {
        // faces[refs[groupref[0]]].adjacent_refs.as_mut().map(|a| a.insert(refs[groupref[1]]));
        faces[refs[groupref[0]]]
            .adjacent_refs
            .insert(refs[groupref[1]]);
        faces[refs[groupref[1]]]
            .adjacent_refs
            .insert(refs[groupref[0]]);
        /*
        if let Some(ref mut a) = faces[refs[groupref[0]]].adjacent_refs {
            a.insert(refs[groupref[1]]);
        }
        if let Some(ref mut a) = faces[refs[groupref[1]]].adjacent_refs {
            a.insert(refs[groupref[0]]);
        }
        */
    }
} */

#[cfg(test)]
mod tests_solid {
    use super::*;
    // use approx::assert_abs_diff_eq;
    #[test]
    pub fn solid_validation() {
        let mut solid = Solid::new();
        solid.name = "cube".to_string();
        solid.dimension = 2;
        // let eq = vec![1.0, 0.0, 0.7];
        let f1 = Face::new_from_halfspace(&vec![1.0, 0.0, 0.7]);
        let f2 = Face::new_from_halfspace(&vec![-1.0, 0.0, 0.7]);
        let f3 = Face::new_from_halfspace(&vec![0.0, 2.0, 1.4]);
        let f4 = Face::new_from_halfspace(&vec![0.0, -3.0, 2.1]);
        solid.suffix_face(&f1);
        solid.suffix_face(&f2);
        solid.suffix_face(&f3);
        solid.suffix_face(&f4);
        // println!("precompute {}", solid.export_to_json(JsonFormat::Pretty));

        solid.compute_adjacencies();
        // println!("postcompute {}", solid.export_to_json(JsonFormat::Pretty));
        /*  let json_face = r#"{
                       "face": [ 1, 0, 0, 0, -0.049999999999999684 ]
                   }"#;
        */
        /* let json_solid_4d = r##"{
            "solidname": "cube",
            "dimension": 4,
            "color": "#aF099F",
            "faces": [ {
                "face": [ 1, 0, 0, 0, -0.05 ]
            }, {
                "face": [ -1, 0, 0, 0, 0.55 ]
            }, {
                "face": [ 0, 1, 0, 0, -0.45 ]
            }, {
                "face": [ 0, -1, 0, 0, 0.95 ]
            }, {
                "face": [ 0, 0, 1, 0, 0.3 ]
            }, {
                "face": [ 0, 0, -1, 0, 0.2 ]
            }, {
                "face": [ 0, 0, 0, 1, -0.15 ]
            }, {
                "face": [ 0, 0, 0, -1, 0.65 ]
            } ]
        }"##; */

        let json_solid = r##"{
            "solidname": "2D test",
            "dimension": 2,
            "color": "#aF099F",
            "faces": [ {
                "face": [ 1, 1, -5]
            }, {
                "face": [ 1, -1, 3]
            }, {
                "face": [ 0, 1, -1]
            }, {
                "face": [ -1, 0, 6 ]
            } ]
        }"##;

        // let parsed_face: JsonFace = serde_json::from_str(&json_face).unwrap();

        /*
            println!(" {:?} ", parsed_face);
            println!(" {:?} ", parsed_face.face);
            println!(" {:?} ", parsed_face.face[0]);
        */
        // let parsed_solid: JsonSolid = serde_json::from_str(&json_solid).unwrap();

        //  println!(" {:?} ", parsed_solid);
        //  println!(" {:?} ", parsed_solid.color);
        //  println!(" {:?} ", parsed_solid.faces);

        let mut sol = dbg!(<Solid>::new_from_json(json_solid.to_string()));

        // println!("sol {:?} ", sol);
        dbg!(&sol);
        // sol.compute(ComputeMode::Basic);
        sol.compute(ComputeMode::Full);
        // sol.compute_adjacencies();

        // println!("ensure faces {:?} ", sol);
        dbg!(&sol);
        /*
            assert_eq!(sol.corners.len(), 4);
            assert_eq!(
                sol.faces.iter().all(|f| f.touching_corners.len() == 2),
                true
            );
        */
        sol.translate(&vec![-3.0, -5.0]);
        // on vérifie la réinitialisation des éléments avec la translation
        assert_eq!(sol.corners.len(), 0);
        assert_eq!(
            sol.faces.iter().all(|f| f.touching_corners.len() == 0),
            true
        );
        // dbg!(&sol);

        // sol.compute_adjacencies_secured();
        sol.compute(ComputeMode::Full);

        dbg!(&sol);

        let proj_sol = sol.project(&Axis(0), ProjecType::Projection);

        dbg!(&proj_sol);

        /*
            assert_eq!(sol.corners.len(), 4);
            assert_eq!(
                sol.faces.iter().all(|f| f.touching_corners.len() == 2),
                true
            );

            dbg!(&sol);

            sol.translate(&vec![2.0, 3.0]);
            sol.compute(ComputeMode::Full);
        */

        // dbg!(sol);

        /*
        if let Some(mut p0) = sol.project(1, ProjecType::Projection) {
           // p0.compute(ComputeMode::Basic);

            assert_eq!(p0.corners.len(), 2);
        } else {
            dbg!("erreur de projection");
            assert_eq!(1, 2);
        }
        */
        /*
            let mut sol4d = dbg!(<Solid>::new_from_json(json_solid_4d.to_string()));
            sol4d.compute(ComputeMode::Full);

            assert_eq!(sol4d.corners.len(), 16);

            if let Some(mut p0) = sol4d.project(1, ProjecType::Projection) {
                p0.compute(ComputeMode::Basic);
             //   dbg!(&p0);
                assert_eq!(p0.corners.len(), 8);
            } else {
                dbg!("erreur de projection");
                assert_eq!(1, 2);
            }
        */
        // dbg!(sol4d.export_to_json(JsonFormat::Export));

        // sol.compute_adjacencies();

        // println!("ensure faces {:?} ", sol);
        // dbg!(&sol4d);

        /*

        */
    }
}
