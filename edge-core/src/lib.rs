#![allow(non_snake_case)]

pub mod config;
pub mod ring_buffer;
pub mod rotation;
pub mod trajectory;

pub use config::MidasConfig;
pub use ring_buffer::RingBuffer;
pub use rotation::{
    budget_gated_rotation, project_to_manifold, rotate_manifold_givens, rotation_angle,
};
pub use trajectory::{compute_momentum, penetration_epsilon};
