# How to create a custom metric
1. Create a subclass in custom_metrics.py called `{YourMetric}` which inherits from the `CustomMetric` class. See the `ExactMatchDiffMetric` class in src/metrics/custom_metrics.py for an example.
2. Implement the init function and give the metric a unique name.
3. Implement the function evaluate returns a score for a list of examples.
5. Add the new class to `CUSTOM_METRICS` in src/utils.py
