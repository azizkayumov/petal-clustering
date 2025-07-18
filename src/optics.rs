use std::collections::HashMap;
use std::ops::{AddAssign, DivAssign};

use ndarray::{Array, ArrayBase, Data, Ix2};
use num_traits::{float::FloatCore, FromPrimitive};
use petal_neighbors::{
    distance::{Euclidean, Metric},
    BallTree,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::Fit;

/// OPTICS (ordering points to identify the clustering structure) clustering
/// algorithm.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use petal_neighbors::distance::Euclidean;
/// use petal_clustering::{Optics, Fit};
///
/// let points = array![[1., 2.], [2., 5.], [3., 6.], [8., 7.], [8., 8.], [7., 3.]];
/// let clustering = Optics::new(4.5, 2, Euclidean::default()).fit(&points, None);
///
/// assert_eq!(clustering.0.len(), 2);        // two clusters found
/// assert_eq!(clustering.0[&0], [0, 1, 2]);  // the first three points in Cluster 0
/// assert_eq!(clustering.0[&1], [3, 4, 5]);  // the rest in Cluster 1
/// ```
#[derive(Debug, Deserialize, Serialize)]
pub struct Optics<A, M> {
    /// The radius of a neighborhood.
    pub eps: A,

    /// The minimum number of points required to form a dense region.
    pub min_samples: usize,

    /// The metric to compute distance between the entries.
    pub metric: M,

    ordered: Vec<usize>,
    reachability: Vec<A>,
    neighborhoods: Vec<Neighborhood<A>>,
}

impl<A> Default for Optics<A, Euclidean>
where
    A: FloatCore,
{
    fn default() -> Self {
        Self {
            eps: A::from(0.5_f32).expect("valid float"),
            min_samples: 5,
            metric: Euclidean::default(),
            ordered: vec![],
            reachability: vec![],
            neighborhoods: vec![],
        }
    }
}

impl<A, M> Optics<A, M>
where
    A: FloatCore,
    M: Metric<A>,
{
    #[must_use]
    pub fn new(eps: A, min_samples: usize, metric: M) -> Self {
        Self {
            eps,
            min_samples,
            metric,
            ordered: vec![],
            reachability: vec![],
            neighborhoods: vec![],
        }
    }

    #[must_use]
    pub fn extract_clusters_and_outliers(
        &self,
        eps: A,
    ) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
        let mut outliers = vec![];
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();

        for &id in &self.ordered {
            if self.reachability[id].is_normal() && self.reachability[id] <= eps {
                if clusters.is_empty() {
                    outliers.push(id);
                } else {
                    let Some(v) = clusters.get_mut(&(clusters.len() - 1)) else {
                        unreachable!("`clusters` is not empty and its key is a sequence number");
                    };
                    v.push(id);
                }
            } else {
                let n = &self.neighborhoods[id];
                if n.neighbors.len() >= self.min_samples && n.core_distance <= eps {
                    clusters.entry(clusters.len()).or_insert_with(|| vec![id]);
                } else {
                    outliers.push(id);
                }
            }
        }
        (clusters, outliers)
    }
}

/// Fits the OPTICS clustering algorithm to the given input data.
///
/// # Parameters
/// - `input`: A 2D array representing the dataset to cluster. Each row corresponds to a data point.
/// - `_params`: An optional parameter for prelabelled data. Not used in this implementation, but required for consistency.
///
/// # Returns
/// A tuple containing:
/// - `HashMap<usize, Vec<usize>>`: A mapping of cluster IDs to the indices of points in each cluster.
/// - `Vec<usize>`: A vector of indices representing the noise points that do not belong to any cluster.
impl<S, A, M>
    Fit<ArrayBase<S, Ix2>, HashMap<usize, Vec<usize>>, (HashMap<usize, Vec<usize>>, Vec<usize>)>
    for Optics<A, M>
where
    A: AddAssign + DivAssign + FloatCore + FromPrimitive + Send + Sync,
    S: Data<Elem = A> + Sync,
    M: Metric<A> + Clone + Sync,
{
    fn fit(
        &mut self,
        input: &ArrayBase<S, Ix2>,
        _params: Option<&HashMap<usize, Vec<usize>>>,
    ) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
        if input.is_empty() {
            return (HashMap::new(), vec![]);
        }

        self.neighborhoods = if input.is_standard_layout() {
            build_neighborhoods(input, self.eps, self.metric.clone())
        } else {
            let input = Array::from_shape_vec(input.raw_dim(), input.iter().copied().collect())
                .expect("valid shape");
            build_neighborhoods(&input, self.eps, self.metric.clone())
        };
        let mut visited = vec![false; input.nrows()];
        self.ordered = Vec::with_capacity(input.nrows());
        self.reachability = vec![A::nan(); input.nrows()];
        for (idx, n) in self.neighborhoods.iter().enumerate() {
            if visited[idx] || n.neighbors.len() < self.min_samples {
                continue;
            }
            process(
                idx,
                input,
                self.min_samples,
                &self.metric,
                &self.neighborhoods,
                &mut self.ordered,
                &mut self.reachability,
                &mut visited,
            );
        }
        self.extract_clusters_and_outliers(self.eps)
    }
}

#[allow(clippy::too_many_arguments)]
fn process<S, A, M>(
    idx: usize,
    input: &ArrayBase<S, Ix2>,
    min_samples: usize,
    metric: &M,
    neighborhoods: &[Neighborhood<A>],
    ordered: &mut Vec<usize>,
    reachability: &mut [A],
    visited: &mut [bool],
) where
    A: FloatCore,
    S: Data<Elem = A>,
    M: Metric<A>,
{
    let mut to_visit = vec![idx];
    while let Some(cur) = to_visit.pop() {
        if visited[cur] {
            continue;
        }
        visited[cur] = true;
        ordered.push(cur);
        if neighborhoods[cur].neighbors.len() < min_samples {
            continue;
        }
        let mut seeds = vec![];
        update(
            cur,
            &neighborhoods[cur],
            input,
            visited,
            metric,
            &mut seeds,
            reachability,
        );
        while let Some(s) = seeds.pop() {
            if visited[s] {
                continue;
            }
            visited[s] = true;
            ordered.push(s);
            let n = &neighborhoods[s];
            if n.neighbors.len() < min_samples {
                continue;
            }
            update(
                s,
                &neighborhoods[s],
                input,
                visited,
                metric,
                &mut seeds,
                reachability,
            );
        }
    }
}

fn update<S, A, M>(
    id: usize,
    neighborhood: &Neighborhood<A>,
    input: &ArrayBase<S, Ix2>,
    visited: &[bool],
    metric: &M,
    seeds: &mut Vec<usize>,
    reachability: &mut [A],
) where
    A: FloatCore,
    S: Data<Elem = A>,
    M: Metric<A>,
{
    for &o in &neighborhood.neighbors {
        if visited[o] {
            continue;
        }
        let reachdist = reachability_distance(o, id, input, neighborhood, metric);
        if !reachability[o].is_normal() {
            reachability[o] = reachdist;
            seeds.push(o);
        } else if reachability[o].lt(&reachdist) {
            reachability[o] = reachdist;
        }
    }
    seeds.sort_unstable_by(|a, b| {
        reachability[*a]
            .partial_cmp(&reachability[*b])
            .unwrap()
            .reverse()
    });
}

#[derive(Debug, Deserialize, Serialize)]
struct Neighborhood<A> {
    pub neighbors: Vec<usize>,
    pub core_distance: A,
}

fn build_neighborhoods<S, A, M>(
    input: &ArrayBase<S, Ix2>,
    eps: A,
    metric: M,
) -> Vec<Neighborhood<A>>
where
    A: AddAssign + DivAssign + FloatCore + FromPrimitive + Send + Sync,
    S: Data<Elem = A>,
    M: Metric<A> + Sync,
{
    if input.nrows() == 0 {
        return Vec::new();
    }
    let rows: Vec<_> = input.rows().into_iter().collect();
    let db = BallTree::new(input.view(), metric).expect("non-empty array");
    rows.into_par_iter()
        .map(|p| {
            let neighbors = db.query_radius(&p, eps).into_iter().collect::<Vec<usize>>();
            let core_distance = if neighbors.len() > 1 {
                db.query(&p, 2).1[1]
            } else {
                A::zero()
            };
            Neighborhood {
                neighbors,
                core_distance,
            }
        })
        .collect()
}

fn reachability_distance<S, A, M>(
    o: usize,
    p: usize,
    input: &ArrayBase<S, Ix2>,
    neighbors: &Neighborhood<A>,
    metric: &M,
) -> A
where
    A: FloatCore,
    S: Data<Elem = A>,
    M: Metric<A>,
{
    let dist = metric.distance(&input.row(o), &input.row(p));
    if dist.gt(&neighbors.core_distance) {
        dist
    } else {
        neighbors.core_distance
    }
}

#[cfg(test)]
mod test {
    use maplit::hashmap;
    use ndarray::{array, aview2};

    use super::*;

    #[test]
    fn default() {
        let optics = Optics::<f32, Euclidean>::default();
        assert_eq!(optics.eps, 0.5);
        assert_eq!(optics.min_samples, 5);
    }

    #[test]
    fn optics() {
        let data = array![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];

        let mut model = Optics::new(0.5, 2, Euclidean::default());
        let (mut clusters, mut outliers) = model.fit(&data, None);
        outliers.sort_unstable();
        for (_, v) in clusters.iter_mut() {
            v.sort_unstable();
        }

        assert_eq!(hashmap! {0 => vec![0, 1, 2, 3], 1 => vec![4, 5]}, clusters);
        assert_eq!(Vec::<usize>::new(), outliers);
    }

    #[test]
    fn core_samples() {
        let data = array![[0.], [2.], [3.], [4.], [6.], [8.], [10.]];
        let mut model = Optics::new(1.01, 1, Euclidean::default());
        let (clusters, outliers) = model.fit(&data, None);
        assert_eq!(clusters.len(), 5); // {0: [0], 1: [1, 2, 3], 2: [4], 3: [5], 4: [6]}
        assert!(outliers.is_empty());
    }

    #[test]
    fn fit_empty() {
        let data: Vec<[f64; 8]> = vec![];
        let input = aview2(&data);

        let mut model = Optics::new(0.5, 2, Euclidean::default());
        let (clusters, outliers) = model.fit(&input, None);
        assert!(clusters.is_empty());
        assert!(outliers.is_empty());
    }
}
