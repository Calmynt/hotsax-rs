<h1 align="center">hotsax</h1>

<div align="center">
<sub>
An implementation of the HOTSAX discord discovery algorithm.
</sub>
</div>

<br/>

<div align="center">
  <a href="https://crates.io/crates/hotsax">
    <img src="https://img.shields.io/crates/v/hotsax.svg" alt="hotsax crate">
  </a>
   <a href="https://docs.rs/crate/hotsax">
    <img src="https://docs.rs/hotsax/badge.svg" alt="hotsax docs">
  </a>
</div>

<br/>

{{readme}}

## Evaluation
To show the accuracy of the implementation, the algorithm was run on the same
dataset as used in the paper itself. Specifically, data from Figure 6 and Figure 7
(as can be retrieved [here](https://www.cs.ucr.edu/~eamonn/discords/), or from the `data/`
directory of this repository as `TEK16.CSV` and `TEK17.CSV` respectively.

The algorithm was ran with a word size of 3, an alphabet size of 3, and a discord size of 128.

Below show the results of this algorithm, compared with the figures shown in the paper.

![Figure 6](./imgs/img1-keogh.png)
![`hotsax` on `TEK16.CSV`](./imgs/img1-ours.png)
![Figure 7](./imgs/img2-keogh.png)
![`hotsax` on `TEK17.CSV`](./imgs/img2-ours.png)