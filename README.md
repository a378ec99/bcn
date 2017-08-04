# bcn
**Blind Compressive Normalization**

The algorithm recovers bias from a measurement matrix without requiring quantitative standards (based on detectable redundancies in the data alone).

**Dependencies**

- anaconda
- pymanopt
- mpi4py (for cluster implementation)

**Abstract**

Blind compressive normalization of public high-throughput databases

Motivation: The rise of high-throughput technologies in the domain of molecular and cell biology has generated an unprecedented amount of quantitative high-dimensional data. Public databases at present make a wealth of this data available, but appropriate normalization is critical for meaningful analysis across different experiments and technologies. Because of missing experimental annotation and lack of quantitative standards, large-scale normalization across entire databases is currently limited to approaches that demand ad hoc assumptions about noise sources and biological signal. Without appropriate normalization, meta-analyses are moot and so is the potential to address shortcomings in experimental designs with public data.

Results: By leveraging detectable redundancies in public databases, such as related samples and features, we show that blind normalization of confounding factors is possible. The proposed approach is formulated in the theoretical framework of compressed sensing and uses manifold optimization for the recovery of bias. As database sizes increase more complex bias can be normalized. In addition, our approach accounts for missing values and can incorporate side information, such as spike-ins. We highlight the potential application of blind compressive normalization to large high-throughput databases and evaluate its performance and robustness in simulation. This work presents the first systematic approach to the *post hoc* removal of bias in high-throughput databases.
