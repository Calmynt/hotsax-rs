use num::Float;
use std::ops::{AddAssign, Deref, Index, RangeBounds};
use super::trie::AugmentedTrie;
use std::collections::{HashMap, HashSet};
use super::util::{znorm, gaussian};
use rand::seq::SliceRandom;
use std::slice::SliceIndex;
use std::ops::Bound::*;

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
pub struct Keogh<'a, N: Float> {
    data: &'a Vec<N>,
    discord_size: usize,
    sax_word_length: usize,
    alpha: usize,
    use_brute_force: bool,
    index: _Index,
}

impl<'a, N: Float> Keogh<'a, N> {

    /// Sets up the data and the discord size to be used.
    pub fn with(data: &'a Vec<N>, discord_size: usize) -> Self {
        Self {
            data,
            discord_size,
            sax_word_length: 3,
            alpha: 3,
            use_brute_force: false,
            index: _index_from_range(.., data.len())
        }
    }

    /// Sets a flag to use the brute force algorithm instead of the HOT SAX one.
    pub fn use_brute_force(&mut self) -> &mut Self {
        self.use_brute_force = true;
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

        let discord = if self.use_brute_force {
            KeoghInternal::brute_force_best(
                &use_subslice,
                self.discord_size
            )
        }
        else {
            KeoghInternal::get_top_discord(
                &use_subslice,
                self.discord_size,
                self.sax_word_length,
                self.alpha
            )
        };

        discord.map(|(dist, loc)| (dist, loc+self.index.0))
    }

    /// Finds the top `n` largest discords. The vector returned can have *less* than `n` elements
    /// if less than `n` discords could be found.
    pub fn find_n_largest_discords(&self, discord_amnt: usize) -> Vec<(f64, usize)> {
        let use_subslice = self.data.get(self.index.0..self.index.1).unwrap();

        let discords = if self.use_brute_force {
            KeoghInternal::brute_force_top_n(
                &use_subslice,
                self.discord_size,
                discord_amnt
            )
        }
        else {
            KeoghInternal::get_top_n_discords(
                &use_subslice,
                self.discord_size,
                self.sax_word_length,
                self.alpha,
                discord_amnt
            )
        };

        discords.into_iter().map(|(dist, loc)| (dist, loc+self.index.0)).collect()
    }
}

/// Separates the internal logic of the algorithm from the public API.
struct KeoghInternal {}

impl KeoghInternal {
    /// Brute force algorithm for finding discords. Made private due to substandard performance.
    ///
    /// Incredibly accurate, but slow to execute. Always takes n^2 time.
    fn brute_force_internal<N, R>(
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

    pub fn brute_force_top_n<N, R>(
        data: &R,
        discord_size: usize,
        discord_amnt: usize
    ) -> Vec<(f64, usize)> where N: Float, R: Deref<Target=[N]> {
        let mut discords = Vec::new();
        let mut skip_over = Vec::new();

        loop {
            let discord = KeoghInternal::brute_force_internal(
                data,
                discord_size,
                &[]
            );

            if discord.0 == 0.0 {
                break discords
            }

            discords.push(discord);

            if discords.len() >= discord_amnt {
                break discords
            }

            skip_over.extend(discord.1-discord_size..discord.1+discord_size);
        }
    }

    #[inline]
    pub fn brute_force_best<N, R>(
        data: &R,
        discord_size: usize
    ) -> Option<(f64, usize)> where N: Float, R: Deref<Target=[N]> {
        KeoghInternal::brute_force_top_n(data, discord_size, 1).pop()
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
    pub fn get_top_n_discords<N, R>(
        data: &R,
        discord_size: usize,
        sax_word_length: usize,
        alpha: usize,
        discord_amnt: usize
    ) -> Vec<(f64, usize)> where N: Float, R: Deref<Target=[N]> {
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
        // `word_table`
        let word_table = KeoghInternal::attach_freq_sax_words(&words)
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

        let mut discords : Vec<(f64, usize)> = Vec::new();
        let mut skip_over = Vec::new();

        loop {
            let discord = KeoghInternal::hot_sax_internal(
                &word_table,
                &trie,
                discord_size,
                &znorm,
                skip_over.as_slice()
            );

            if discord.0 == 0.0 {
                break discords
            }

            discords.push(discord);

            if discords.len() >= discord_amnt {
                break discords
            }

            skip_over.extend(discord.1-discord_size..discord.1+discord_size);
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
    pub fn get_top_discord<N, R>(
        data: &R,
        discord_size: usize,
        sax_word_length: usize,
        alpha: usize
    ) -> Option<(f64, usize)> where N: Float, R: Deref<Target=[N]> {
        KeoghInternal::get_top_n_discords(data, discord_size, sax_word_length, alpha,  1).pop()
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
    fn hot_sax_internal<N, R>(
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
}