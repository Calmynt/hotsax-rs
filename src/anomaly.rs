use num::Float;
use std::ops::{AddAssign, Deref, Index, RangeBounds};
use super::trie::AugmentedTrie;
use std::collections::{HashMap, HashSet};
use super::util::{znorm, gaussian};
use rand::seq::SliceRandom;
use std::slice::SliceIndex;
use std::ops::Bound::*;
use crate::paa;

struct _Index(usize, usize);

fn _index_from_range(bounds: impl RangeBounds<usize>, len: usize) -> _Index {
    let startbound = bounds.start_bound();
    let endbound = bounds.end_bound();
    let x = if let Included(x) = startbound {*x} else {0};
    let y = if let Excluded(x) = endbound {*x}
    else if let Included(x) = endbound {x+1}
    else {len};
    _Index(x,y)
}

/// Provides easy access to the HOT SAX and brute force algorithms, by using the builder pattern.
/// The only necessary algorithmic parameter is the size of the discord itself. The rest have
/// default values, namely:
/// - `sax_word_length` = 3
/// - `alpha` = 3
/// - `use_brute_force` = false
pub struct Anomaly<'a, N: Float> {
    data: &'a Vec<N>,
    discord_size: usize,
    sax_word_length: usize,
    alpha: usize,
    algo: Algorithm,
    dim_reduce: usize,
    index: _Index,
}

pub enum Algorithm {
    Bruteforce,
    HOTSAX,
    Squeezer(f64)
}

impl<'a, N: Float> Anomaly<'a, N> {

    /// Sets up the data and the discord size to be used.
    ///
    /// By default it uses:
    /// - `sax_word_length: 3`
    /// - `alpha: 3`
    /// - `algo: Algorithm::HOTSAX`
    /// - `dim_reduce: 0` (disabled)
    pub fn with(data: &'a Vec<N>, discord_size: usize) -> Self {
        Self {
            data,
            discord_size,
            sax_word_length: 3,
            alpha: 3,
            algo: Algorithm::HOTSAX,
            dim_reduce: 0,
            index: _index_from_range(.., data.len())
        }
    }

    /// Determines the exact algorithm to use.
    pub fn use_algo(&mut self, algo: Algorithm) -> &mut Self {
        self.algo = algo;
        self
    }

    /// Sets the length of the SAX words to use.
    pub fn sax_word_length(&mut self, n: usize) -> &mut Self {
        self.sax_word_length = n;
        self
    }

    /// Specifies to only use a slice of the dataset.
    pub fn use_slice(&mut self, range: impl RangeBounds<usize>) -> &mut Self {
        self.index = _index_from_range(range, self.data.len());
        self
    }

    /// Applies PAA to the data. This is only useful if run with
    /// the bruteforce algorithm in mind.
    pub fn dim_reduce(&mut self, new_len: usize) -> &mut Self {
        self.dim_reduce = new_len;
        self
    }

    /// Sets the alphabet size to be used. The only valid values should be in the range 3..=7.
    ///
    /// ## Panics
    /// - When `n` is set to an invalid value.
    pub fn alpha(&mut self, n: usize) -> &mut Self {
        if (n<3) | (n>7) {
            panic!("Invalid setting for alphabet size ({}). Only values in 3-7 are supported.", n);
        }

        self.alpha = n;
        self
    }

    /// Finds the largest discord. If one couldn't be found, this function returns a `None` instead.
    pub fn find_largest_discord(&self) -> Option<(f64, usize)> {
        let use_subslice = self.data.get(self.index.0..self.index.1).unwrap();

        let discord = match self.algo {
            Algorithm::Bruteforce => {
                if self.dim_reduce > 1 {
                    anomaly_internal::brute_force_best(
                        &paa(&use_subslice.to_vec(), self.dim_reduce),
                        self.discord_size
                    ).map(|(dist, loc)| (dist, loc*((1000/self.dim_reduce) as usize)))
                } else {
                    anomaly_internal::brute_force_best(
                        &use_subslice,
                        self.discord_size
                    )
                }
            },
            Algorithm::HOTSAX => {
                anomaly_internal::hotsax_best(
                    &use_subslice,
                    self.discord_size,
                    self.sax_word_length,
                    self.alpha
                )
            },
            Algorithm::Squeezer(threshold) => {
                anomaly_internal::hs_squeezer_best(
                    &use_subslice,
                    self.discord_size,
                    self.sax_word_length,
                    self.alpha,
                    threshold
                )
            }
        };

        discord.map(|(dist, loc)| (dist, loc+self.index.0))
    }

    /// Finds the top `n` largest discords. The vector returned can have *less* than `n` elements
    /// if less than `n` discords could be found.
    pub fn find_n_largest_discords(&self, discord_amnt: usize) -> Vec<(f64, usize)> {
        let use_subslice = self.data.get(self.index.0..self.index.1)
            .expect(&format!("Couldn't retrieve subslice ({}..{})", self.index.0, self.index.1));

        let discords = match self.algo {
            Algorithm::Bruteforce => {
                if self.dim_reduce > 1 {
                    anomaly_internal::brute_force_top_n(
                        &paa(&use_subslice.to_vec(), self.dim_reduce),
                        self.discord_size,
                        discord_amnt
                    ).into_iter().map(|(dist, loc)| (dist, loc*((1000/self.dim_reduce) as usize))).collect()
                } else {
                    anomaly_internal::brute_force_top_n(
                        &use_subslice,
                        self.discord_size,
                        discord_amnt
                    )
                }
            },
            Algorithm::HOTSAX => {
                anomaly_internal::hotsax_top_n(
                    &use_subslice,
                    self.discord_size,
                    self.sax_word_length,
                    self.alpha,
                    discord_amnt
                )
            },
            Algorithm::Squeezer(threshold) => {
                anomaly_internal::hs_squeezer_top_n(
                    &use_subslice,
                    self.discord_size,
                    self.sax_word_length,
                    self.alpha,
                    threshold,
                    discord_amnt
                )
            }
        };

        discords.into_iter().map(|(dist, loc)| (dist, loc+self.index.0)).collect()
    }

    /// Finds all discords with a measured distance above `min_dist`.
    pub fn find_discords_min_dist(&self, min_dist: f64) -> Vec<(f64, usize)> {
        let use_subslice = self.data.get(self.index.0..self.index.1).unwrap();

        let discords = match self.algo {
            Algorithm::Bruteforce => {
                if self.dim_reduce > 1 {
                    anomaly_internal::brute_force_min_dist(
                        &paa(&use_subslice.to_vec(), self.dim_reduce),
                        self.discord_size,
                        min_dist
                    ).into_iter().map(|(dist, loc)| (dist, loc*((1000/self.dim_reduce) as usize))).collect()
                } else {
                    anomaly_internal::brute_force_min_dist(
                        &use_subslice,
                        self.discord_size,
                        min_dist
                    )
                }
            },
            Algorithm::HOTSAX => {
                anomaly_internal::hotsax_min_dist(
                    &use_subslice,
                    self.discord_size,
                    self.sax_word_length,
                    self.alpha,
                    min_dist
                )
            },
            Algorithm::Squeezer(threshold) => {
                anomaly_internal::hs_squeezer_min_dist(
                    &use_subslice,
                    self.discord_size,
                    self.sax_word_length,
                    self.alpha,
                    threshold,
                    min_dist
                )
            }
        };

        discords.into_iter().map(|(dist, loc)| (dist, loc+self.index.0)).collect()
    }
}


// Internal algorithms for anomaly detection. To be called by Keogh.
mod anomaly_internal {
    use num::Float;
    use std::ops::Deref;
    use crate::anomaly::{keogh_util, inner_algo};
    use crate::znorm;

    pub fn brute_force_top_n<N, R>(
        data: &R,
        discord_size: usize,
        discord_amnt: usize
    ) -> Vec<(f64, usize)> where N: Float, R: Deref<Target=[N]> {
        let mut discords = Vec::new();
        let mut skip_over = Vec::new();

        loop {
            let discord = inner_algo::brute_force_internal(
                data,
                discord_size,
                &skip_over
            );

            if discord.0 == 0.0 {
                break discords
            }

            discords.push(discord);

            if discords.len() >= discord_amnt {
                break discords
            }

            let min = 0.max(discord.1 as isize - discord_size as isize) as usize;
            skip_over.extend(min..discord.1+discord_size);
        }
    }

    pub fn brute_force_min_dist<N, R>(
        data: &R,
        discord_size: usize,
        min_dist: f64,
    ) -> Vec<(f64, usize)> where N: Float, R: Deref<Target=[N]> {
        let mut discords = Vec::new();
        let mut skip_over = Vec::new();

        loop {
            let discord = inner_algo::brute_force_internal(
                data,
                discord_size,
                &[]
            );

            if (discord.0 == 0.0) | (discord.0 < min_dist) {
                break discords
            }

            discords.push(discord);

            let min = 0.max(discord.1 as isize - discord_size as isize) as usize;
            skip_over.extend(min..discord.1+discord_size);
        }
    }

    #[inline]
    pub fn brute_force_best<N, R>(
        data: &R,
        discord_size: usize
    ) -> Option<(f64, usize)> where N: Float, R: Deref<Target=[N]> {
        brute_force_top_n(data, discord_size, 1).pop()
    }

    /// The HOT SAX algorithm as proposed by Keogh et al.
    ///
    /// Accurate, and faster than the brute force algorithm. Takes between `n` and `n^2` time.
    ///
    /// ## Panics
    /// - `sax_word_length` is larger than `discord size`.
    /// - `alpha` is under 3 or over 7.
    ///
    /// ## Returns
    /// A list of the distances of the top n discords (0), as well as their locations. (1)
    /// This list can have less elements if less discords were found.
    pub fn hotsax_top_n<N, R>(
        data: &R,
        discord_size: usize,
        sax_word_length: usize,
        alpha: usize,
        discord_amnt: usize
    ) -> Vec<(f64, usize)> where N: Float, R: Deref<Target=[N]> {
        let words = keogh_util::get_sax_words(data, discord_size, sax_word_length, alpha);
        let (word_table, trie, znorm) = keogh_util::extract_hotsax_items(data, &words);

        let mut discords : Vec<(f64, usize)> = Vec::new();
        let mut skip_over = Vec::new();

        loop {
            let discord = inner_algo::hot_sax_internal(
                &word_table,
                &trie,
                discord_size,
                &znorm,
                &skip_over
            );

            if discord.0 == 0.0 {
                break discords
            }

            discords.push(discord);

            if discords.len() >= discord_amnt {
                break discords
            }

            let min = 0.max(discord.1 as isize - discord_size as isize) as usize;
            skip_over.extend(min..discord.1+discord_size);
        }
    }

    pub fn hotsax_min_dist<N, R>(
        data: &R,
        discord_size: usize,
        sax_word_length: usize,
        alpha: usize,
        min_dist: f64,
    ) -> Vec<(f64, usize)> where N: Float, R: Deref<Target=[N]> {
        let words = keogh_util::get_sax_words(data, discord_size, sax_word_length, alpha);
        let (word_table, trie, znorm) = keogh_util::extract_hotsax_items(data, &words);

        let mut discords : Vec<(f64, usize)> = Vec::new();
        let mut skip_over = Vec::new();

        loop {
            let discord = inner_algo::hot_sax_internal(
                &word_table,
                &trie,
                discord_size,
                &znorm,
                &skip_over
            );

            if (discord.0 == 0.0) | (discord.0 < min_dist) {
                break discords
            }

            discords.push(discord);

            let min = 0.max(discord.1 as isize - discord_size as isize) as usize;
            skip_over.extend(min..discord.1+discord_size);
        }
    }

    #[inline]
    /// The HOT SAX algorithm as proposed by Keogh et al. As suggested by the paper, the alphabet
    /// size used is hardcoded as `3`.
    ///
    /// Accurate, and faster than the brute force algorithm. Takes between `n` and `n^2` time.
    ///
    /// A shortcut function to `get_top_n_discords(..., 1)`
    ///
    /// ## Panics
    /// `sax_word_length` is larger than `discord size`.
    ///
    /// ## Returns
    /// The distance of the best discord (0), as well as its location. (1)
    ///
    /// If such a discord isn't found, this function returns `None`.
    pub fn hotsax_best<N, R>(
        data: &R,
        discord_size: usize,
        sax_word_length: usize,
        alpha: usize
    ) -> Option<(f64, usize)> where N: Float, R: Deref<Target=[N]> {
        hotsax_top_n(data, discord_size, sax_word_length, alpha, 1).pop()
    }

    /// The HS-Squeezer algorithm.
    ///
    /// Should be approximately 4x faster than HOT SAX.
    ///
    /// ## Panics
    /// - `sax_word_length` is larger than `discord size`.
    /// - `alpha` is under 3 or over 7.
    ///
    /// ## Returns
    /// A list of the distances of the top n discords (0), as well as their locations. (1)
    /// This list can have less elements if less discords were found.
    pub fn hs_squeezer_top_n<N, R>(
        data: &R,
        discord_size: usize,
        sax_word_length: usize,
        alpha: usize,
        threshold: f64,
        discord_amnt: usize
    ) -> Vec<(f64, usize)> where N: Float, R: Deref<Target=[N]> {
        let words = keogh_util::get_sax_words(data, discord_size, sax_word_length, alpha);
        let znorm = znorm(data);

        let mut discords : Vec<(f64, usize)> = Vec::new();
        let mut skip_over = Vec::new();

        loop {
            let discord = inner_algo::hs_squeezer_internal(
                &words,
                discord_size,
                &znorm,
                threshold,
                &skip_over
            );

            if discord.0 == 0.0 {
                break discords
            }

            discords.push(discord);

            if discords.len() >= discord_amnt {
                break discords
            }

            let min = 0.max(discord.1 as isize - discord_size as isize) as usize;
            skip_over.extend(min..discord.1+discord_size);
        }
    }

    /// The HS-Squeezer algorithm.
    ///
    /// Should be approximately 4x faster than HOT SAX.
    ///
    /// ## Panics
    /// - `sax_word_length` is larger than `discord size`.
    /// - `alpha` is under 3 or over 7.
    ///
    /// ## Returns
    /// A list of the distances of all discords above the min_dist (0), as well as their locations. (1)
    /// This list can have less elements if less discords were found.
    pub fn hs_squeezer_min_dist<N, R>(
        data: &R,
        discord_size: usize,
        sax_word_length: usize,
        alpha: usize,
        threshold: f64,
        min_dist: f64,
    ) -> Vec<(f64, usize)> where N: Float, R: Deref<Target=[N]> {
        let words = keogh_util::get_sax_words(data, discord_size, sax_word_length, alpha);
        let znorm = znorm(data);

        let mut discords : Vec<(f64, usize)> = Vec::new();
        let mut skip_over = Vec::new();

        loop {
            let discord = inner_algo::hs_squeezer_internal(
                &words,
                discord_size,
                &znorm,
                 threshold,
                &skip_over
            );

            if (discord.0 == 0.0) | (discord.0 < min_dist) {
                break discords
            }

            discords.push(discord);

            let min = 0.max(discord.1 as isize - discord_size as isize) as usize;
            skip_over.extend(min..discord.1+discord_size);
        }
    }

    #[inline]
    /// The HOT SAX algorithm as proposed by Keogh et al. As suggested by the paper, the alphabet
    /// size used is hardcoded as `3`.
    ///
    /// Accurate, and faster than the brute force algorithm. Takes between `n` and `n^2` time.
    ///
    /// A shortcut function to `get_top_n_discords(..., 1)`
    ///
    /// ## Panics
    /// `sax_word_length` is larger than `discord size`.
    ///
    /// ## Returns
    /// The distance of the best discord (0), as well as its location. (1)
    ///
    /// If such a discord isn't found, this function returns `None`.
    pub fn hs_squeezer_best<N, R>(
        data: &R,
        discord_size: usize,
        sax_word_length: usize,
        alpha: usize,
        threshold: f64
    ) -> Option<(f64, usize)> where N: Float, R: Deref<Target=[N]> {
        hs_squeezer_top_n(data, discord_size, sax_word_length, alpha, threshold, 1).pop()
    }
}

// Implementations of algorithms, with parameters given.
mod inner_algo {
    use crate::trie::AugmentedTrie;
    use num::Float;
    use std::ops::Deref;
    use std::collections::HashSet;
    use crate::gaussian;
    use rand::seq::SliceRandom;
    use crate::squeezer::{Cluster, squeezer};

    /// Brute force algorithm for finding discords. Made private due to substandard performance.
    ///
    /// Incredibly accurate, but slow to execute. Always takes n^2 time.
    pub fn brute_force_internal<N, R>(
        data: &R,
        n: usize,
        skip_over: &[usize]
    ) -> (f64, usize) where N: Float, R: Deref<Target=[N]> {
        let mut best_dist = 0.0;
        let mut best_loc = 0;

        for i in 0..data.len()-n+1 {
            if skip_over.contains(&i) { continue }
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

    /// An internal function that performs the hot sax discord discovery algorithm.
    ///
    /// ## Parameters
    /// - `sorted_word_table` : A word table that's sorted according to the HOT SAX algorithm.
    /// - `word_trie` : An `AugmentedTrie` that represents the sorted word table.
    /// - `alpha` : The alphabet size.
    /// - `discord_size` : The size of the discords to be found.
    /// - `znorm_data` : The data.
    /// - `skip_over` : A list of indexes to skip over.
    pub fn hot_sax_internal<N, R>(
        sorted_word_table: &Vec<(usize, (&String, usize))>,
        word_trie: &AugmentedTrie,
        discord_size: usize,
        data: &R,
        skip_over: &[usize]
    ) -> (f64, usize) where N: Float, R: Deref<Target=[N]> {
        // The actual discord discovery.
        let mut best_dist = 0.0;
        let mut best_loc = 0;

        // Outer loop heuristic: Uses sorted word table.
        for (i,(word,_)) in sorted_word_table.into_iter() {
            if skip_over.contains(i) {
                continue
            }

            // Other occurrences of the same SAX word using the word trie.
            let occurrences: HashSet<usize> = word_trie.get_indexes(word).clone().into_iter().collect();

            // Boolean that checks whether to perform the random search
            let mut do_random_search = true;

            // The neighbouring distance for the inner loop
            let mut neigh_dist = std::f64::INFINITY;

            // Inner loop heuristic: Checks the occurrences of the same SAX word using the word trie.
            for j in occurrences.into_iter() {
                if (*i as isize - j as isize).abs() >= discord_size as isize {
                    // Retrieves the gaussian distance between to slices
                    let dist = gaussian(&data[*i..*i+ discord_size -1], &data[j..j+ discord_size -1]).to_f64().unwrap();
                    // Updates the neighbouring distance
                    neigh_dist = neigh_dist.min(dist);
                    // Stops searching if a distance word than `best_dist` was found
                    if dist < best_dist { do_random_search = false; break;}
                }
            }

            if !do_random_search { continue }

            // Gets all indexes and shuffles them
            // This includes the occurrences as
            let mut nums: Vec<usize> = (0..data.len()- discord_size +1).collect();
            nums.shuffle(&mut rand::thread_rng());

            // Calculates the closest neighbouring distance
            for j in nums.into_iter() {
                if (*i as isize - j as isize).abs() >= discord_size as isize {
                    let dist = gaussian(&data[*i..*i + discord_size - 1], &data[j..j + discord_size - 1]).to_f64().unwrap();
                    neigh_dist = neigh_dist.min(dist);
                    if dist < best_dist { break; }
                }
            }

            // Updates the best distance if the neighbouring distance is larger.
            if (neigh_dist > best_dist) & (neigh_dist < std::f64::INFINITY) {
                best_dist = neigh_dist;
                best_loc = *i;
            }
        }

        (best_dist, best_loc)
    }

    pub fn hs_squeezer_internal<N, R>(
        words: &Vec<String>,
        discord_size: usize,
        data: &R,
        threshold: f64,
        skip_over: &[usize]
    ) -> (f64, usize) where N: Float, R: Deref<Target=[N]> {
        // The actual discord discovery.
        let mut best_dist = 0.0;
        let mut best_loc = 0;

        // Uses squeezer algorithm to get clusters.
        let clusters = squeezer(&words, threshold);

        let mut indexes = clusters.iter().min_by_key(|cluster| cluster.len()).unwrap().vec();
        indexes.append(&mut (0..data.len()).collect());

        // Outer loop heuristic: Uses sorted word table.
        for i in indexes.into_iter() {
            if skip_over.contains(&i) {
                continue
            }

            // Boolean that checks whether to perform the random search
            let mut do_random_search = true;

            // The neighbouring distance for the inner loop
            let mut neigh_dist = std::f64::INFINITY;

            // Finds the cluster that the current item is in.
            let curr_cluster = if let Some(cluster) = clusters
                .iter()
                .find(|cluster| cluster.contains(&i)) {
                cluster
            } else {
                continue;
            };

            // Inner loop heuristic: Checks the occurrences of the same SAX word using the word trie.
            for &j in curr_cluster.iter() {
                if (i as isize - j as isize).abs() >= discord_size as isize {
                    // Retrieves the gaussian distance between to slices
                    let dist = gaussian(&data[i..i+ discord_size -1], &data[j..j+ discord_size -1]).to_f64().unwrap();
                    // Updates the neighbouring distance
                    neigh_dist = neigh_dist.min(dist);
                    // Stops searching if a distance word than `best_dist` was found
                    if dist < best_dist { do_random_search = false; break;}
                }
            }

            if !do_random_search { continue }

            // Gets all indexes and shuffles them
            // This includes the occurrences as
            let mut nums: Vec<usize> = (0..data.len()- discord_size +1).collect();
            nums.shuffle(&mut rand::thread_rng());

            // Calculates the closest neighbouring distance
            for j in nums.into_iter() {
                if (i as isize - j as isize).abs() >= discord_size as isize {
                    let dist = gaussian(&data[i..i + discord_size - 1], &data[j..j + discord_size - 1]).to_f64().unwrap();
                    neigh_dist = neigh_dist.min(dist);
                    if dist < best_dist { break; }
                }
            }

            // Updates the best distance if the neighbouring distance is larger.
            if (neigh_dist > best_dist) & (neigh_dist < std::f64::INFINITY) {
                best_dist = neigh_dist;
                best_loc = i;
            }
        }

        (best_dist, best_loc)
    }
}

// Utilities used by algorithms, for generating certain parameters.
mod keogh_util {
    use std::ops::{Deref, AddAssign};
    use num::Float;
    use crate::trie::AugmentedTrie;
    use std::collections::HashMap;
    use crate::znorm;
    use rand::seq::SliceRandom;

    pub fn attach_freq_sax_words(words: &Vec<String>) -> Vec<(&String, usize)> {
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

    pub fn extract_hotsax_items<'a, N, R>(
        data: &R,
        words: &'a Vec<String>
    ) -> (Vec<(usize, (&'a String, usize))>, AugmentedTrie, Vec<N>) where N: Float, R: Deref<Target=[N]> {
        let znorm = znorm(data);

        let trie = AugmentedTrie::from_words(words.iter().enumerate().collect());

        // Contains (index, (SAXword, frequency))
        // The former is useful to iterate over the data in an ordered way.
        // The latter is useful for the magic inner loop.
        // `word_table`
        let word_table = attach_freq_sax_words(&words)
            .into_iter()
            .enumerate()
            .collect::<Vec<(usize, (&String, usize))>>();

        // Gets the minimum frequency from the word table
        let min_freq = (word_table.iter().min_by_key(|elem| (elem.1).1).unwrap().1).1;

        // Splits word table into minfreq and the rest.
        let (mut word_table, mut other) : (_,Vec<_>)  = word_table
            .into_iter()
            .partition(|elem| (elem.1).1 == min_freq);

        // Randomly shuffles the rest, then adds them to the word table.
        other.shuffle(&mut rand::thread_rng());
        word_table.extend(other);

        (word_table, trie, znorm)
    }

    pub fn get_sax_words<N, R>(
        data: &R,
        discord_size: usize,
        sax_word_length: usize,
        alpha: usize,
    ) -> Vec<String> where N: Float, R: Deref<Target=[N]> {
        let mut words: Vec<String> = Vec::new();

        for i in 0..data.len() - discord_size {
            words.push(crate::dim_reduction::sax(&data[i..i+ discord_size].to_vec(), sax_word_length, alpha));
        }

        words
    }
}