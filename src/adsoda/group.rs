/**
 * @file Describes ADSODA group
 * @author Jeff Bigot <jeff@raktres.net> after Greg Ferrar
 * @class Group
 * @todo use it
 */

use serde::{Deserialize, Serialize};
use serde_json;

use super::face::{self, Face};
use super::halfspace::{self, among_index, vector_flip};
use super::parameters::{JsonFormat, MAX_FACE_PER_CORNER, VERY_SMALL_NUM};
use super::point::{self, Point};
use uuid::Uuid;

///
/// 
/// attention, un groupe peut contenir un autre groupe
/// 
/// il faut donc que la liste d'objet puisse contenir des groupes ou des solides
/// 
///  
/// 
/// 
/// 
/// 
/// 
/// 
/// 

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)] // Copy, Eq,
pub struct Solid {
    name: String,
    id: usize,
    space: String,
    color: String,
    selected: bool,
    object_list: Vec<Face>,
    uuid: Uuid,
}


/* 
    this.name = name || 'nDobject'
    this.id = 0
    // TODO: utiliser les mêmes couleurs que Three
    this.color = color || 0x000000
    this.selected = false // TODO vérifier
*/

class Group extends NDObject {
  constructor (objects) {
    super('Group')

    this.space = ''
    this.objectList = new Set()
    this.uuid = uuidv4()
    if (objects) this.objectList = objects
  }

impl Group {
    pub fn new() -> Self {
        let dimension = 0;
        let name = "group".to_string();
        let id = 0;
        let color = "000".to_string();
        let selected = false; // Point::new(vec![0.0,0.0,0.0]);

        Solid {
            dimension,
            id: 0,
            name,
            color,
            selected,
            uuid: Uuid::new_v4(),
        }
    }

  /**
   * @returns JSON face description
   */
  pub fn export_to_json () -> string {
     // return `{ "group" : ${JSON.stringify(this.objectList)} }`
  }

  /**
   *
   * @param {*} json
   */
  static importFromJSON (json, space) {
    const grp = new Group()
    const solids = new Map()
    space.solids.forEach(sol => {
      if (sol.id) {
        solids.set(sol.id, sol.uuid)
      }
    })
    json.refs.forEach(sol => {
      const solUuid = solids.get(sol)
      grp.objectList.add(solUuid)
    })
    grp.space = space
    return grp
  }

  /**
   * @returns text face description
   */
  logDetail () {
    return `Group name : ${this.name} \n --- objects : ${
      this.objectList
    } \n `
  }

  /**
   *
   */
  emptyGroup () {
    this.objectList.length = 0
  }

  /**
   * translate the face following the given vector.<br>
   * Translation doesn't change normal vector, Just the constant term need to be changed.
   * new constant = old constant - dot(normal, vector)<br>
   * @param {*} vector the vector indicating the direction and distance to translate this halfspace.
   * @todo vérifrie que mutation nécessaire
   * @returns face this
   */
  pub fn translate (&self, vector: Vec<f64>) {
    // TODO: add selected control
    this.objectList.forEach(idx => {
      const object = this.space.solids.get(idx)
      object.translate(vector)
    })
    return this
  }

  /**
   *  This method applies a matrix transformation to this Halfspace.
   *  @param {matrix} matrix the matrix of the transformation to apply.
   * @todo vérifrie que mutation nécessaire
   * @returns face this
   */
  pub fn transform (&self, matrix: Vec<Vec<f64>>, center: Point) {
    this.objectList.forEach(idx => {
      const object = this.space.solids.get(idx)
      object.transform(matrix, center)
    })
    return this
  }

  /**
   * @todo write
   */
  pub fn middle_of (&self) {
    const dim = this.space.dimension
    const minCorner = []
    const maxCorner = []
    this.objectList.forEach(idx => {
      const object = this.space.solids.get(idx)
      object.corners.forEach(corner => {
        for (let i = 0; i < dim; i++) {
          minCorner[i] = Math.min(corner[i], minCorner[i] || corner[i])
          maxCorner[i] = Math.max(corner[i], maxCorner[i] || corner[i])
        }
      })
    })
    const corners = []
    for (let i = 0; i < dim; i++) {
      corners[i] = (maxCorner[i] + minCorner[i]) / 2
    }
    return corners
  }

  /**
   *
   */
  pub fn add_object (&self, solid: Solid) {
    this.objectList.push(obj)
  }

  /**
   *
   */
  removeObject (obj) {
    // TODO:
  }
}
export { Group }