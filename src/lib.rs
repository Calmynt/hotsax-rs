//! This crate contains an implementation of the HOT SAX algorithm, and
//! the brute force algorithm, as proposed by [Keogh et al.](http://www.cse.cuhk.edu.hk/~adafu/Pub/icdm05time.pdf).
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
//! ```
//! use std::error::Error;
//! use plotly::{Plot, Scatter};
//!
//! # fn main() -> Result<(), Box<dyn Error>> {
//! // Parses the CSV file from the dataset.
//! let mut rdr = csv::ReaderBuilder::new()
//!     .trim(csv::Trim::All)
//!     .from_path("data/RESP_FIG9.CSV")?;
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
//! let discord = hotsax::Keogh::with(&data, discord_size)
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
pub use anomaly::Keogh;

/// Dimensionality reduction techniques.
///
/// Used in the implementation of `HOTSAX`, but can be used externally as well.
pub mod dim_reduction;
pub use dim_reduction::{paa, sax};

/// Miscellaneous utility functions.
pub mod util;
pub use util::{gaussian, znorm, mean, std_dev};

pub(crate) mod trie;