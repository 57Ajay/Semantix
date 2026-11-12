# Statistics refresher

- Mean is sensitive to outliers; the median is not. For latency, report the median and
  the 95th or 99th percentile, never just the mean.
- Variance is the average squared distance from the mean; standard deviation is its
  square root, in the original units.
- A p-value is the probability of seeing data at least this extreme if the null
  hypothesis were true. It is not the probability that the hypothesis is true.
- With small samples, use a t-test rather than a z-test.
- Correlation does not imply causation; check for a third variable.
- For A/B tests, decide the sample size before starting and do not stop early when the
  result looks good.
