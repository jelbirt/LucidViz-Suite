//! Point shape — same as sphere LOD 0.

use super::Vertex;

pub fn build(_lod: super::Lod) -> (Vec<Vertex>, Vec<u32>) {
    // Points always use the lowest-detail sphere regardless of LOD.
    super::sphere::build(super::Lod::Low)
}
