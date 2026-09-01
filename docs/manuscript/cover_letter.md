# Cover letter, Remote Sensing of Environment

Dear Editors,

Please consider our manuscript "Mapping intra-urban social deprivation from Sentinel
imagery using only municipal aggregate labels" for publication in Remote Sensing of
Environment.

The study asks how much neighborhood-scale detail can be recovered when the only
training signal is the aggregate statistics a country already publishes. Mexico
publishes its social deprivation grade at both municipal and census-tract level, which
allowed a calibration that the literature has lacked: models trained purely on municipal
aggregates, validated against 61,430 fully held-out urban tracts, with an oracle trained
on the tract labels bounding what the frozen features support. Weak supervision recovers
84% of that upper bound on test cities opened exactly once, and the supervision
efficiency analysis (number of aggregates, label granularity, sensor and resolution) is
reported on both splits.

Three methodological results should interest this audience beyond the application.
Attention-based multiple-instance learning, the standard weakly supervised localizer,
produces chance-level maps for a reason we isolate and fix by learning from label
proportions. Model selection under weak supervision is close to uninformative about map
quality, which affects any remote sensing pipeline whose product is a map. And a
land-cover baseline ties the foundation-model features within municipalities, locating
the value of learned representations in bag-level prediction rather than intra-urban
ordering. Negative results are reported at the same level of evidence as positive ones.

All data are public, the pipeline is released, and a frozen benchmark (features, labels,
splits, protocol) accompanies the submission so every number is reproducible without the
satellite pipeline.

The manuscript is not under consideration elsewhere.

Sincerely,
Dagoberto Pulido Arias
Independent researcher, Mexico
