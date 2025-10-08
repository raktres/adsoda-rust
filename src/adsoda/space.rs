//! Solid
//!
//!
// use std::collections::btree_set::Intersection;

use serde::{Deserialize, Serialize};
use tree_ds::prelude::*;
use serde_json;

// use super::*;

use super::bases::{Axis, JsonFormat, Point, ProjecType};
use super::solid::{Solid, NdSolid, ComputeMode};
use uuid::Uuid;

use std::collections::HashMap;

/**
 * @file Describes ADSODA space
 * @author Jeff Bigot <jeff@raktres.net> after Greg Ferrar
 * @class Space
 */
/// 
/// Structure pour l'export en JSON
/// 
#[derive(Serialize, Deserialize, Debug)]
pub struct JsonSpace {
    spacename: String,
    dimension: usize,
    solids: Vec<Solid>,
    // structure: Tree,
}


pub struct Group {
  name: String,
}

impl Group {
    pub fn new(name: String) -> Self {
        // let spacename = "space".to_string();
        // let dimension = dimension;
        // let remove_hidden = false;

        Group {
            name
        }
    }
  }

/// # Space
/// composé de solide.
/// voir comment ont les regroupes.
/// 
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)] // Copy, Eq,
pub struct Space {
    spacename: String,
    dimension: usize,
    solids: HashMap<Uuid, Solid>, // this.solids = new Map()
    // groups: Map, // this.groups = new Map()
    projections: Vec<Space>, // this.projection = []
    projectype: ProjecType,
    structure: Tree<u128,Uuid>,
    // remove_hidden: bool,
}

impl Space {
    pub fn new(dimension: usize, spacename: String) -> Self {
        // let spacename = "space".to_string();
        // let dimension = dimension;
        // let remove_hidden = false;

        Space {
            spacename,
            dimension,
            solids: HashMap::new(),
            // groups: Map:new(),
            // #[serde(skip_serializing)]
            projections: Vec::<Space>::new(),
            projectype: ProjecType::Projection,
            structure: Tree::new(Some("structure")),
            // #[serde(skip_serializing)]
            // remove_hidden: false
        }
    }

    fn new_from_jsonspace(jsonspace: JsonSpace) -> Self {
        let spacename = jsonspace.spacename;
        // let idx= 0;
        let dimension = jsonspace.dimension;
        // let remove_hidden = false;

        Space {
            spacename,
            // idx,
            dimension,
            solids: HashMap::new(),
            // groups: Map:new(),
            // #[serde(skip_serializing)]
            projections: Vec::<Space>::new(),
            projectype: ProjecType::Projection,
            structure: Tree::new(Some("structure"))
        }
    }

    pub fn export_to_json(&self, format: JsonFormat) -> String {
        let res: String;
        match format {
            JsonFormat::Export { .. } => {
                let jss = JsonSpace {
                    spacename: self.spacename.clone(),
                    dimension: self.dimension,
                    solids: self.solids.values().map(|v| v.clone()).collect(),
                    // groups: "groups".to_string(), // color: self.color.clone(),
                                                  // faces: self.faces.iter().map(|f| f.to_jsonface()).collect(),
                };

                res = serde_json::to_string(&jss).unwrap_or("".to_string());
            }
            JsonFormat::Pretty { .. } => {
                res = serde_json::to_string_pretty(self).unwrap();
            }
            JsonFormat::Condensed { .. } => {
                res = serde_json::to_string(self).unwrap();
            }
        }
        res
    }

    pub fn new_from_json(json: String) -> Self {
        // let parsed_solid =
        match serde_json::from_str(&json) {
            Ok(parsed_space) => Space::new_from_jsonspace(parsed_space),
            Err(_e) => Space::new(0, "err".to_string()),
        }
    }
    /*
        pub fn import_from_json (json) {
        const space = new Space(parseInt(json.dimension), json.spacename)
        ;[...json.solids].forEach(solid => {
          space.suffixSolid(Solid.importFromJSON(solid))
        })
        if (json.groups) {
          ;[...json.groups].forEach(agroup => {
            const group = Group.importFromJSON(agroup, space)
            space.suffixGroup(group)
          })
        }
        // space.groups
        // TODO: ajouter groups
        return space
      }
    */
    // logDetail () {}

    ///
    ///
    //  pub fn suffix_group (&self, grp: Group) {
    //    self.groups.set(grp.uuid, grp)
    //  }

    ///
    pub fn add_solid(&mut self, solid: Solid) {
      let uuid = solid.uuid;
      self.solids.insert(uuid, solid);
      let root = self.structure.add_node( Node::<AutomatedId, Uuid>::new_with_auto_id(Some(uuid)), None).unwrap();
      // TODO: ajouter à la structure
    }

    ///
    pub fn clear_solids(&mut self) {
        self.solids.clear();
        // TODO: vider la structure
        
    }

    ///
    pub fn compute_solids(&mut self) {
        self.solids.retain(|_uuid, solid| {
            solid.compute(ComputeMode::Full);
            true
        });
    }
    // eliminateEmptySolids () { const solids = [...this.solids].filter(solid => solid.isNonEmptySolid()); this.solids = solids   }

    /// TODO: voir usage
    pub fn trig_solid(&self, uuid: Uuid) { 
      // TODO: activer dans structure
        /*
        let found_in_group = false;
        this.groups.forEach(grp => {
          if (grp.objectList.has(uuid)) {
            foundInGroup = true
            console.log('found in group')
            grp.selected = !grp.selected
            grp.objectList.forEach(solUuid => {
              const solid = this.solids.get(solUuid)
              console.log('add solid', solUuid, solid )
              solid.selected = !solid.selected
            })
          }
        })
        if (!foundInGroup) {
          const solid = this.solids.get(uuid)
          if (solid) {
            console.log('add solid', uuid, solid )
            solid.selected = !solid.selected
          }
        }
        */
    }

    /// TODO!
    ///
    pub fn transform(&mut self, matrix: Vec<Vec<f64>>, center: Point) {
        // if (this.groups.size === 0) {
        dbg!(self.all_selected());
        // TODO: gérer la structure
        // 
        self.solids.retain(|_uuid, solid| {
          solid.transform(&matrix, &center);
          true
        });
        /*

            this.groups.forEach(grp => {
              if (grp.selected || allSelected) {
                done = true
                const centerptg = center ? grp.middleOf() : [0, 0, 0, 0]
                grp.transform(matrix, centerptg, force)
              }
            })
            if (!done) {
              this.solids.forEach(solid => {
                if (allSelected || solid.selected) {
                  const centerpt = center ? solid.middleOf() : [0, 0, 0, 0]
                  solid.transform(matrix, centerpt, force)
                }
              })
            }
        */

        /*  } else {
          this.groups.forEach(group => {
            const centerptg = center ? group.middleOf() : [0, 0, 0, 0]
            group.transform(matrix, centerptg, force)
          })
        }
        */
    }

    /**
     *
     * @param {*} vector
     * @param {*} force
     */
    pub fn translate(&mut self, vector: Vec<f64>) {
        // TODO: gérer la structure
        // if (this.groups.size === 0) {

        // TODO: test all_selected
        dbg!(self.all_selected());
        self.solids.retain(|_uuid, solid| {
          solid.translate(&vector);
          true
        });
    }

    fn all_selected(&self) -> bool {
        /*
            this.solids.forEach(solid => {
          if (solid.selected) { allSelected = false }
        })
         */
        true
    }
    ///
    ///  TODO: est-ce utile ?
    /// @param {*} axe for the moment, just the index of axe
    ///  @returns return an array of solids in which hidden parts are removed
    ///  amène de la complexité pour un résultat à vérifier
    ///
    // pub fn remove_hidden_solids (&self, axe)


    /// project solids from space following axe
    pub fn fill_projections(&mut self) {
        self.projections = Vec::<Space>::new();
        let dim = self.dimension;
        for i in 0..dim {
            let spacename: String;

            match self.projectype {
                ProjecType::Cut => spacename = " cut axis ".to_string() + &i.to_string(),
                ProjecType::Projection => spacename = "proj axis ".to_string() + &i.to_string(),
            }

            let mut proj_space = Space::new(dim - 1, spacename);
            self.solids.retain(|_uuid, solid| {
                match solid.project(&Axis(i), self.projectype.clone()) {
                    Some(s) => proj_space.add_solid(s),
                    None => (),
                }
                true
            });
            self.projections.push(proj_space);
        }
    }

    // Cut solids
    //pub fn get_cut_solids(&self, axis: &Axis) {
        /*
        const cuts = []
        this.solids.forEach((val, key) => cuts.push(val.axeCut(axe)))
        const solids = cuts.reduce((solflat, item) => solflat.concat(item), [])
          .filter(solid => solid.isNonEmptySolid())
        return solids
        */
    // }

    // Project space following axe
    // * @param {*} axe for the moment, just the index of axe
    //  @returns space

    // pub fn get_proj_space(&self, axis: &Axis) -> Space {
    //  return self.projections[axis.0]

        /* let projectype = self.projectype.clone();
        let space = Space::new(&self.dimension - 1, self.project_name(axis));
        self.solids.retain(|uuid, solid| {
            // solid.compute(ComputeMode::Full);
            //todo!() ;
            // for i in 0..dim {
            space
                .solids
                .insert(solid.get_silhouette(Axis(i), self.projectype));
            // }
            true
        }); */
        /*
        const solidarray = this.projectSolids(axe)
        solidarray.forEach(solid => space.suffixSolid(solid))
        return space
        */
    // }

    // Project space following axe
     
    // pub fn get_cut_space(&self, axis: Axis) {
        /*    // if (REMOVE_HIDDEN)
        const space = new Space(this.dimension - 1)
        space.name = this.axeCutName(axe)
        // TODO il faut que project solids utilise filteredSolids
        const solidarray = this.axeCutSolids(axe)
        solidarray.forEach(solid => space.suffixSolid(solid))
        return space */
    // }
}

mod tests_solid {
    use super::*;
    // use approx::assert_abs_diff_eq;
    #[test]
    pub fn solid_validation() {
    }
  }
