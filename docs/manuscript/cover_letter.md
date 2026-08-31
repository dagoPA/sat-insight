# Cover letter — Nature Communications

Dear Editors,

Please consider our manuscript, "Mapping neighborhood deprivation from municipal
statistics and satellite imagery", for publication in Nature Communications.

Nearly every country publishes social statistics as administrative aggregates; almost
none publishes them at neighborhood level. Our study measures, for the first time
against a fully held-out neighborhood-level ground truth at national scale, how much
neighborhood detail those aggregates recover when combined with free satellite
imagery. Mexico publishes its deprivation index at both municipal and census-tract
level, which lets us calibrate the exchange rate between the two: models trained only on
municipal aggregates reach 84–90% of the tract-level ordering that the same satellite
features support under full supervision. The calibration axes are operational — how many
aggregates, at what granularity, from which sensor — and the result is priced in policy
units through a targeting simulation.

Three findings extend beyond our study system. Attention-based multiple-instance
learning, the standard weakly supervised localizer, produces maps at chance for a
structural reason we isolate. Model selection under weak supervision is close to
uninformative about map quality, which affects any pipeline whose product is the map,
computational pathology included. And the learned deprivation gradient transfers
zero-shot to Bogotá's block-level strata while informality detection in Rio does not
follow, separating two constructs the literature often conflates.

The test cities were opened exactly once, after all development ended, and every
headline number carries its confirmatory test value. All data are public; the frozen
benchmark (features, labels, splits, protocol) accompanies the submission.

Suggested referees: researchers in satellite-based welfare measurement (the
Jean/Yeh/Chi lineage), weakly supervised learning, and small-area estimation.

The manuscript is not under consideration elsewhere. A preprint will be posted to arXiv
upon submission.

Sincerely,
Dagoberto Pulido Arias
Independent researcher, Mexico
