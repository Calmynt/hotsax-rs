//! This crate contains an implementation of the HOT SAX algorithm, and
//! the brute force algorithm, as proposed by [Keogh et al.](http://www.cse.cuhk.edu.hk/~adafu/Pub/icdm05time.pdf).
//! It will also include the [HS-Squeezer](https://dl.acm.org/doi/abs/10.1145/3287921.3287929) algorithm when it is implemented,
//! since it offers useful optimizations, while still being heavily based on the HOT SAX algorithm.
//!
//! During the implementation some other functions had to be made, such as `paa`, `znorm`, and
//! `gaussian`. These functions are exposed, due to their utility apart from being used in HOT SAX.
//!
//! The code is well commented in order to explain the implementation, in the case that people want
//! to learn how the HOT SAX algorithm works by looking at an implementation. If a part is vaguely
//! commented, feel free to leave an issue.
//!
//! Note that only `Float` vectors are supported. If your data is made up of integers, you need to
//! convert it to float first.
//!
//! ## Example of use
//! ```ignore
//! use std::error::Error;
//! use plotly::{Plot, Scatter};
//!
//! # fn main() -> Result<(), Box<dyn Error>> {
//! // Parses the CSV file from the dataset.
//! let mut rdr = csv::ReaderBuilder::new()
//!     .trim(csv::Trim::All)
//!     .from_path("data/TEK16.CSV")?;
//!
//! // Deserialize CSV data into a vector of floats.
//! let mut data : Vec<f64> = Vec::new();
//! for result in rdr.deserialize() {
//!     data.push(result?);
//! }
//!
//! // Prepare a plot
//! let mut plot = Plot::new();
//!
//! // Retrieve the largest discord. This should approx. match the one found in the paper.
//! // It uses the same settings: a discord size of 256 and a=3.
//! // word_size was assumed to be 3.
//! let discord_size = 256;
//! let discord = hotsax::Anomaly::with(&data, discord_size)
//!     .use_slice(1000..)      // Skips the beginning due to an abnormality.
//!     .find_largest_discord() // Finds the largest discord in the subslice.
//!     .unwrap().1;            // Only gets the location.
//!
//! // Plot the entire dataset as a blue color.
//! let trace1 = Scatter::new((1..=data.len()).collect(), data.clone())
//!     .line(plotly::common::Line::new().color(plotly::NamedColor::Blue))
//!     .name("Data");
//!
//! // Plot the discord itself as a red color.
//! let trace2 = Scatter::new((discord+1..discord+discord_size+1).collect(), data[discord..discord+128].to_vec())
//!     .line(plotly::common::Line::new().color(plotly::NamedColor::Red))
//!     .name("Discord");
//!
//! // Add traces to the plot.
//! plot.add_trace(trace1);
//! plot.add_trace(trace2);
//!
//! // Shows the plot to verify.
//! plot.show();
//! # Ok(())
//! # }
//! ```

/// Implements anomaly detection algorithms, including the brute force and
/// HOT SAX algorithms as specified by Keogh's paper, found
/// [here](http://www.cse.cuhk.edu.hk/~adafu/Pub/icdm05time.pdf).
pub mod anomaly;
pub use anomaly::Anomaly;

/// Dimensionality reduction techniques.
///
/// Used in the implementation of `HOTSAX`, but can be used externally as well.
pub mod dim_reduction;
pub use dim_reduction::{paa, sax};

/// Miscellaneous utility functions.
pub mod util;
pub use util::{gaussian, znorm, mean, std_dev};

/// Clustering functions and squeezer impl
pub mod squeezer;
pub use squeezer::squeezer;
pub use anomaly::Algorithm;

pub(crate) mod trie;

#[cfg(test)]
mod test {
    use plotly::{Plot, Scatter, Layout};

    static DISCORD_SIZE: usize = 128;
    static DISCORD_AMNT: usize = 1;
    static MIN_DIST: f64 = 2.00;

    #[test]
    fn multi_discord() -> Result<(), Box<dyn std::error::Error>> {

        // Initialises the CSV reader.
        let mut rdr = csv::ReaderBuilder::new()
            .trim(csv::Trim::All)
            .from_path("data/TEK17.csv")?;

        // Preparing Y axis...
        let mut data: Vec<f64> = Vec::new();

        // Retrieve all data.
        for record in rdr.deserialize() {
            data.push(record?);
        }

        // let data = trailing_moving_average(&data, 0);

        // Retrieve all discords.
        let discords = crate::Anomaly::with(&data, DISCORD_SIZE)
            .use_algo(crate::Algorithm::Squeezer(0.85))
            // .sax_word_length(5)
            // .dim_reduce(800)
            // .use_slice(4000..)
            // .find_discords_min_dist(MIN_DIST);
            .find_n_largest_discords(DISCORD_AMNT);

        println!("{} DISCORDS FOUND!", discords.len());

        let indexes : Vec<_> = (1..=5000).collect();

        // Initialise plot with original data.
        let mut plot = Plot::new();
        let temps = Scatter::new(indexes.clone(), data.clone()).name("temps");
        plot.add_trace(temps);

        // Plots all discords.
        plot_discords(&mut plot, discords, &indexes, &data);
        // Attaches a title to the plot.
        attach_title(&mut plot, "Temperature 2019, Buchan Sydney");

        // Shows the plot.
        plot.show();

        Ok(())
    }

    // Plots all discords onto the passed in plot.
    fn plot_discords(plot: &mut Plot, discords: Vec<(f64, usize)>, x_axis: &Vec<u32>, data: &Vec<f64>) {
        for i in 0..discords.len() {
            let discord = discords[i];
            let loc = discord.1;
            let dist = discord.0;

            plot.add_trace(
                Scatter::new(
                    x_axis[loc..loc+DISCORD_SIZE].into(),
                    data[loc..loc+DISCORD_SIZE].into()
                ).name(&format!("Discord #{} ({:.2})", i, dist))
            );
        }
    }

    fn attach_title(plot: &mut Plot, title: &str) {
        plot.set_layout(
            Layout::new()
                .title(plotly::common::Title::new(title))
        );
    }

    /// Averages the data using a centered moving average.
    ///
    /// Data[i] = Mean(Data[i-n..i+n]);
    fn centered_moving_average(data: &Vec<f64>, n: usize) -> Vec<f64> {
        if n == 0 { return data.clone() }

        let mut out = Vec::new();
        let len = data.len();

        for i in 0..n {
            out.push(crate::mean(&&data[0..i]));
        }

        for i in n..len-n {
            out.push(crate::mean(&&data[i-n..i+n]));
        }

        for i in len-n..len {
            out.push(crate::mean(&&data[i-n..i]));
        }

        out
    }

    /// Averages the data using a trailing moving average.
    ///
    /// Data[i] = Mean(Data[i-n..i]);
    fn trailing_moving_average(data: &Vec<f64>, n: usize) -> Vec<f64> {
        if n == 0 { return data.clone() }

        let mut out = Vec::new();
        let len = data.len();

        for i in 0..n {
            out.push(crate::mean(&&data[0..i]));
        }

        for i in n..len {
            out.push(crate::mean(&&data[i-n..i]));
        }

        out
    }
}