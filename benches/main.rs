mod dbscan;
mod optics;

use crate::dbscan::{
    build as dbscan_build, fixed_clusters as dbscan_fixed_clusters,
    uniform_clusters as dbscan_uniform_clusters,
};
use crate::optics::{
    build as optics_build, fixed_clusters as optics_fixed_clusters,
    uniform_clusters as optics_uniform_clusters,
};
use criterion::{criterion_group, criterion_main, Criterion};

criterion_group! {
name = benches;
config = Criterion::default()
    .sample_size(100)
    .measurement_time(std::time::Duration::new(60, 0));
targets =
    dbscan_build, dbscan_fixed_clusters, dbscan_uniform_clusters,
    optics_build, optics_fixed_clusters, optics_uniform_clusters
}

criterion_main!(benches);