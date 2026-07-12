#![allow(non_snake_case)]

pub mod config;
pub mod ring_buffer;
pub mod rotation;
pub mod trajectory;

pub use config::MidasConfig;
pub use ring_buffer::RingBuffer;
pub use rotation::{
    budget_gated_rotation, budget_gated_rotation_fixed_basis, compute_fixed_basis,
    project_to_manifold, rotate_manifold_givens, rotate_manifold_givens_fixed_basis,
    rotation_angle,
};
pub use trajectory::{compute_momentum, penetration_epsilon_windowed};
