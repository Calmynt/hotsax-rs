use num::Float;
use std::ops::{Deref, DerefMut};
use std::slice::Iter;

pub struct Cluster {
    _clusters: Vec<usize>
}

impl Cluster {
    pub fn new (tuple: usize) -> Self {
        Cluster {
            _clusters: vec![tuple]
        }
    }

    pub fn add (&mut self, elem: usize) {
        self._clusters.push(elem);
    }

    pub fn vec (&self) -> Vec<usize> {
        self._clusters.clone()
    }
}

impl Deref for Cluster {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self._clusters
    }
}

impl DerefMut for Cluster {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self._clusters
    }
}

pub fn squeezer(data: &Vec<String>, threshold: f64) -> Vec<Cluster> {
    let mut clusters: Vec<Cluster> = vec![Cluster::new(0)];

    for (i, instance) in data[1..].iter().enumerate() {
        let mut similarity = vec![0.00; clusters.len()];

        for (index, elem) in clusters.iter_mut().enumerate() {
            similarity[index] = similarity_cluster(data, instance, elem);
        }

        let simmax = similarity.iter().cloned().fold(f64::neg_infinity(), |a, b| a.max(b));

        let mut sin_max_cluster_id = 0;

        for j in 0..clusters.len() {
            if similarity[j] == simmax {
                sin_max_cluster_id = j;
            }
        }

        if simmax >= threshold {
            clusters[sin_max_cluster_id].add(i+1);
        } else {
            clusters.push(Cluster::new(i+1));
        }
    }

    clusters
}

fn similarity_cluster(data: &Vec<String>, val: &String, cluster: &Cluster) -> f64 {
    let mut unique = vec![];

    for &elem in cluster.iter() {
        if !unique.contains(&data[elem]) {
            unique.push(data[elem].clone());
        }
    }

    let mut temp = 0;

    for elem in unique.iter() {
        temp += get_support(data, elem, &cluster);
    }

    get_support(data, val, &cluster) as f64 / temp as f64
}

fn get_support(words: &Vec<String>, value: &String, cluster: &Cluster) -> usize{
    let mut num = 0;
    for &index in cluster.iter() {
        if words[index] == value.clone() { num += 1; }
    }
    num
}