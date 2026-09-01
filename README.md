1. Natural Recovery Model

A Random Forest classifier estimates the probability that a customer will recover without intervention.

Features include:

Cart amount
Average order value
Previous purchases
Previous abandonments
Previous reminders
Previous recoveries
Purchase frequency
Cart-to-average-order ratio
Minutes since abandonment
2. Intervention Model

A second Random Forest model estimates recovery probability under intervention.

Keeping this model separate from the natural recovery model allows RecoverAI to estimate the difference between the two potential outcomes.

3. Uplift Estimation

The prototype estimates customer-level uplift by comparing:

Intervention probability
        -
Natural recovery probability

The resulting uplift is converted into expected incremental revenue using cart value.

4. Decision Policy

RecoverAI uses expected incremental revenue to determine whether intervention is worthwhile.

The current prototype uses a configurable ₹500 expected incremental revenue threshold.

The threshold was evaluated against alternative values including:

₹500
₹1,000
₹2,000
₹3,000
₹4,000
₹5,000

The ₹500 threshold produced the highest total expected incremental revenue in the current synthetic evaluation, while also targeting more customers.

Machine Learning

RecoverAI currently uses two Random Forest classifiers:

Natural Recovery Model

Predicts:

P(recovery without intervention)
Intervention Model

Predicts:

P(recovery with intervention)

The two predictions are then compared to estimate uplift.

Random Forest was selected for the prototype because it:

Handles nonlinear relationships
Captures feature interactions
Requires relatively little preprocessing
Is fast to train and iterate
Provides probability estimates for the decision layer
Validation
Unseen Test Set

The natural recovery model was evaluated on a held-out test set of 200 customers.

Current results:

Accuracy: 72.50%
Precision: 40.00%
Recall: 11.54%

The recall limitation is explicitly acknowledged rather than hidden. The prototype prioritizes targeted intervention decisions rather than attempting blanket recovery prediction.

Probability Calibration

A Brier-score calibration experiment was also performed.

The original model achieved:

Brier score: 0.0540

A sigmoid-calibrated version achieved:

Brier score: 0.1835

Since calibration worsened the score on this dataset, the original model was retained.

Uplift Experiment

The model-based uplift experiment currently produces:

Customers targeted: 436
Average estimated uplift: 44.88%
Expected incremental revenue: ₹11,15,569
Simulated incremental revenue: ₹13,77,864

These figures are simulation results, not real revenue generated for Razorpay.

They demonstrate the methodology on the prototype dataset.

Monte Carlo Simulation

Because recovery outcomes are probabilistic, a single simulation can vary.

RecoverAI therefore includes a Monte Carlo experiment using 500 simulations.

Current result:

Average simulated recovered revenue: ₹7,73,618.73
Standard deviation: ₹38,680.91
95% simulation interval:
₹6,99,370.28 – ₹8,44,671.60

This provides a more robust view of simulation variability than relying on a single random outcome.

Data Disclaimer

The current dataset is synthetic.

RecoverAI does not use proprietary Razorpay customer or transaction data.

The synthetic dataset models checkout behavior using variables such as cart value, purchase history, abandonment history, reminders, and time since abandonment.

The purpose of the synthetic dataset is to prototype and evaluate the decision methodology.

Therefore:

Revenue figures shown by the prototype should not be interpreted as real Razorpay revenue or production performance.

Production Path

A production version of RecoverAI would require real checkout and intervention data.

Potential data fields include:

Customer/session identifier
Cart value
Checkout timestamp
Previous purchases
Previous abandonment history
Previous intervention history
Intervention type
Recovery outcome
Transaction value
Consent/communication preferences
Randomized Validation

The most important production improvement would be controlled experimentation.

A randomized holdout could compare:

Control group
No intervention

vs.

Treatment group
RecoverAI-selected intervention

This would allow the system to measure true incremental treatment effect rather than relying on synthetic assumptions.

Future Uplift Modeling

With sufficient randomized data, RecoverAI could evolve toward dedicated causal/uplift approaches such as:

T-Learner
X-Learner
Doubly robust estimation
Causal forests

This would provide a stronger estimate of the actual incremental effect of intervention.

Risk Controls

A production system should also include:

Communication opt-outs
Frequency caps
Consent checks
Intervention cost modeling
Revenue thresholds
Monitoring for model drift
Probability calibration monitoring
Human review for high-value edge cases
Project Structure
RecoverAI/
│
├── backend/
│   ├── app.py
│   ├── recovery.py
│   ├── model.py
│   ├── treatment_model.py
│   ├── recovery_model.pkl
│   ├── treatment_model.pkl
│   ├── evaluate.py
│   ├── calibration_test.py
│   ├── threshold_test.py
│   ├── uplift_test.py
│   ├── uplift_model_test.py
│   ├── monte_carlo_test.py
│   ├── batch_test.py
│   ├── revenue_test.py
│   └── frontend/
│
└── data/
    ├── checkout_data.csv
    └── generate_data.py
Prototype Limitations

RecoverAI is currently a prototype.

The main limitations are:

The dataset is synthetic.
Intervention outcomes are simulated rather than collected from a randomized experiment.
The two-model uplift approach is a prototype rather than a validated causal model.
Model probabilities require further calibration validation on larger real-world datasets.
The current natural recovery model has relatively low recall.
Production deployment would require privacy, consent, communication, and monitoring infrastructure.

These limitations are intentional areas for future development rather than hidden assumptions.

Future Roadmap
Phase 1 — Prototype
Synthetic checkout data
Natural recovery model
Intervention model
Uplift estimation
Revenue-based targeting
Monte Carlo simulation
Dashboard
Phase 2 — Real Data Pilot
Integrate real checkout events
Collect randomized intervention outcomes
Validate probability calibration
Measure true incremental revenue
Add communication-cost modeling
Phase 3 — Production ML
Dedicated causal/uplift modeling
Model monitoring
Automated retraining
Drift detection
Experimentation platform
Compliance and consent controls
Key Philosophy

RecoverAI does not try to contact every abandoned customer.

It tries to identify the customers where intervention has the highest expected incremental value.

Recovery isn't about reaching everyone — it's about knowing who's worth reaching.


