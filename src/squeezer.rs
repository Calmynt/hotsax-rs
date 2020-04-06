use num::Float;
use std::ops::{Deref, DerefMut};

type Cluster = Vec<usize>;

pub fn squeezer(data: &Vec<String>, threshold: f64) -> Vec<Cluster> {
    let mut clusters: Vec<Cluster> = vec![vec![0]];

    for (i, instance) in data[1..].iter().enumerate() {

        let mut similarity = Vec::new();
        for elem in clusters.iter() {
            similarity.push(similarity_cluster(data, instance, elem));
        }

        let simmax = similarity.iter().fold(0.0, |a, b| a.max(*b));

        let sin_max_cluster_id = similarity.iter().position(|&x| x == simmax).unwrap_or(0);

        if simmax >= threshold {
            clusters[sin_max_cluster_id].push(i+1);
        } else {
            clusters.push(vec![i+1]);
        }
    }

    clusters
}

fn similarity_cluster(data: &Vec<String>, val: &String, cluster: &Cluster) -> f64 {
    let mut unique = vec![];
    for &elem in cluster.iter() {
        if !unique.contains(&&data[elem]) {
            unique.push(&data[elem]);
        }
    }

    let mut temp = 0;

    for elem in unique.into_iter() {
        temp += get_support(data, elem, &cluster);
    }

    get_support(data, val, &cluster) as f64 / temp as f64
}

fn get_support(words: &Vec<String>, value: &String, cluster: &Cluster) -> usize{
    let mut num = 0;
    for &index in cluster.iter() {
        if &words[index] == value { num += 1; }
    }
    num
}