use num::Float;
use super::dist::gaussian;
use std::ops::AddAssign;
use super::trie::AugmentedTrie;
use std::collections::HashMap;
use std::process::exit;
use crate::util::znorm;
use rand::seq::SliceRandom;

/// Brute force algorithm for finding discords.
///
/// Incredibly accurate, but slow to execute. Always takes n^2 time.
pub fn brute_force<N>(data: &Vec<N>, n: usize) -> (f64, usize) where N: Float {
    let mut best_dist = 0.0;
    let mut best_loc = 0;

    for i in 0..data.len()-n+1 {
        let mut neigh_dist = std::f64::INFINITY;
        for j in 0..data.len()-n+1 {
            if (i as isize - j as isize).abs() >= n as isize {
                let dist = gaussian(&data[i..i+n-1], &data[j..j+n-1]);
                neigh_dist = neigh_dist.min(dist.to_f64().unwrap());
            }
        }

        if neigh_dist > best_dist {
            best_dist = neigh_dist;
            best_loc = i;
        }
    }

    (best_dist, best_loc)
}

fn attach_freq_sax_words(words: &Vec<String>) -> Vec<(&String, usize)> {
    let mut freqmap: HashMap<&String, usize> = HashMap::new();

    words.iter().for_each(|word| {
        if freqmap.contains_key(word) {
            freqmap.get_mut(word).unwrap().add_assign(1);
        }
        else {
            freqmap.insert(word, 1);
        }
    });

    words.iter().map(|word| {
        (word, freqmap[word])
    }).collect()
}

/// The HOT SAX algorithm as proposed by Keogh et al. As suggested by the paper, the alphabet
/// size used is hardcoded as `3`.
///
/// Accurate, and faster than the brute force algorithm. Takes between `n` and `n^2` time.
///
/// ## Panics
/// `sax_word_length` is larger than `discord size`.
///
/// ## Returns
/// The distance of the best discord (0), as well as its location. (1)
pub fn hot_sax<N>(data: &Vec<N>, discord_size: usize, sax_word_length: usize) -> (f64, usize) where N: Float {
    let alpha = 3;
    let len = data.len();
    let mut words: Vec<String> = Vec::new();

    let znorm = znorm(data);

    for i in 0..len- discord_size {
        words.push(super::dim_reduction::sax(&data[i..i+ discord_size].to_vec(), sax_word_length, alpha));
    }

    let trie = AugmentedTrie::from_words(words.iter().enumerate().collect());

    // Contains (index, (SAXword, frequency))
    // The former is useful to iterate over the data in an ordered way.
    // The latter is useful for the magic inner loop.
    let word_table = attach_freq_sax_words(&words)
        .into_iter()
        .enumerate()
        .collect::<Vec<(usize, (&String, usize))>>();

    let mut sorted_word_table = word_table.clone();

    // Not exactly like HOT SAX, because it's a full sort.
    // TODO: try implementing the true algorithm
    sorted_word_table.sort_by_key(|elem| (elem.1).1);

    // The actual discord discovery.
    let mut best_dist = 0.0;
    let mut best_loc = 0;

    for (i,(word,_)) in sorted_word_table.into_iter() {
        // Other occurrences of the same SAX word
        let occurrences = trie.get_indexes(word).clone();

        // Boolean that checks whether to perform the random search
        let mut do_random_search = true;

        // The neighbouring distance for the inner loop
        let mut neigh_dist = std::f64::INFINITY;

        for j in occurrences.into_iter() {
            if (i as isize - j as isize).abs() >= discord_size as isize {
                // Retrieves the gaussian distance between to slices
                let dist = gaussian(&znorm[i..i+ discord_size -1], &znorm[j..j+ discord_size -1]).to_f64().unwrap();
                // Updates the neighburing distance
                if dist < neigh_dist { neigh_dist = dist };
                // Stops searching if a distance word than `best_dist` was found
                if dist < best_dist { do_random_search = false; break;}
            }
        }

        if do_random_search {
            // Gets all indexes and shuffles them
            let mut nums: Vec<usize> = (0..data.len()- discord_size +1).collect();
            nums.shuffle(&mut rand::thread_rng());

            // Calculates the closest neighbouring distance
            for j in nums.into_iter() {
                if (i as isize - j as isize).abs() >= discord_size as isize {
                    let dist = gaussian(&znorm[i..i + discord_size - 1], &znorm[j..j + discord_size - 1]).to_f64().unwrap();
                    if dist < best_dist {
                        break;
                    }
                    neigh_dist = neigh_dist.min(dist);
                }
            }

            // Updates the best distance if the neighbouring distance is larger.
            if (neigh_dist > best_dist) & (neigh_dist < std::f64::INFINITY) {
                best_dist = neigh_dist;
                best_loc = i;
            }
        }
    }

    (best_dist, best_loc)
}