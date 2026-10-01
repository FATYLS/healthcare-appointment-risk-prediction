# A/B testing framework — SMS reminders

The historical `SMS_received` field is observational. It does not document randomized treatment assignment, so the historical dataset cannot establish that SMS reminders caused a change in no-show rate.

## Proposed experiment
**H0:** the reminder has no effect on no-show rate.

**H1:** the reminder changes the no-show rate.

Randomly assign eligible appointments to a control workflow and a treatment workflow. Pre-register the primary metric, minimum detectable effect, sample size/power, significance level, stopping rule and guardrail metrics. Report absolute effect, relative effect, confidence interval and p-value.

Potential guardrails include cancellation/rescheduling behavior, patient contact complaints and operational load.
