# Original VS Standardised

- For all pairs (forward, backward) non-linear models have higher **baseline scores**


- Pairs where both models have good baseline scores are good models for **reusability** on both sides. These pairs are OCSVM, MMD pairs when data are standardised. When data remain unchanged OCSVM scores need to be high on both sides for this to be true.


- When data are standardised for all experiments **OCSVM** correctly predicts the direction for reuability for the model that yields the vest results the majority of the time. Granted in some cases the opposite direction can have a better model if the correct model is not chosen. However, when data are not standardised this is only true for experiment 1 as for the rest of them it is either partially true or untrue. Therefore OCSVM seems to be more stable when data are standardised


- **MMD** is generally stable except in experiment 3 for non-standardised data

- Differences across experiments: Experiment 1 provides more consistent results for both standardised and non-standardised data. Experiment 2 generally provides consistent results except for model choice which varies more for non-standardised data. Experiment 3 has the most variability in results when comparing standardised to non-standardised data. It is worth mentioning that the sample sizes of experiments decrease and this may be the issue. 

## Per Experiment

### Experiment 1

- **Node baseline models**: pi2, pi3, pi4 (above 0.9 R2) - same for std and non-std data, std: pi5 (just above 0.5) non-std: pi5 (just above 0.3 - samples 1,2,3, just above 0.2 - sample 4)

- Pairs where both models have good baseline scores are good models for reusability on both sides. For non std data this is true when **OCSVM scores are high on both sides 0.57 and higher.**

    A good model when paired with a bad one, reusability results are not good and in this case the "bad" model is always better for reusability but it will never be used since performance will always be much lower than what it could be if we used a native model.s

- **OCSVM, MMD** pairs good for reusability on both sides. For std data: Only pair pi2, pi4 missed by MMD and good for reusability on both sides (OCSVM higher than 0.6 in both sides, not necesarily true for the other two mmd pairs though). For non-std data: Only pair pi3, pi4 missed by MMD and good for reusability on both sides (OCSVM high only on backward side and results are good almost no discrepancy).

- **OCSVM** correctly predicts the direction of for reusability for the model that yields best results, however the opposite direction can have a better model if the correct model is not chosen

#### Model Choice

- For all pairs (forward, backward) non-linear models have higher **baseline scores**

- **std data, std in performance**: with whatever node pi2 and pi4 is paired with the std is small, pi5 has the largest std of all nodes, for node pi3 it depends on the node paired with however, when paired with a node with a good model std remains low. 

- Note on std models: Except when paired with pi5 for the other pair combinations linear models have lower **discrepancy** -> for the pairs that this does hold is not as important that the discrepancy of linear models is smaller since the difference in discrepancy between linear - non-linear models is small.

    ***Therefore for nodes pi2 and pi4 it does not matter too much the kernel choice as it seems it does not affect the discrepancy and therefore model choice should be based on native performance on node. For node pi5 and pi3 the non-linear kernel performs best with all pair combinations.***
    
    
- **non-std data, std in performance**: std varies depending on the pairing. It is not easy to infer the model to choose per node, for example for node pi4 the linear model is better when paired with node pi3 but the non-linear model is better when paired with pi2.

### Experiment 2

- **Node baseline models**  pi2, pi4, pi5 (R2 below 0.5) best model for pi3 is just above 0.6. (samples 1-3: std data, sample 1 non-std data) and just above 0.5 only in sample 4/ samples 2,3,4.


- **OCSVM, MMD** not good reusability if model one of the two models is not good -> since only pi3 has a descent model there are **no good pairs for reusability**


- For all pairs (forward, backward) non-linear models have higher **baseline scores**


- std data: **OCSVM** correctly predicts the direction of for reusability for the model that yields best results, however the opposite direction can have a better model if the correct model is not chosen. OCSVM does not correctly predict the directionality for pair (pi2, pi5) and (pi3, pi5) in samples 2,3 and 4. Similar OCSVM scores yield similar R2 - discrepancy performance (not true for sample 4)

- non-std data: In most **OCSVM** does not correctly predict the direction of for reusability for the model that yields best results. In those cases the models are not good but then in general for the experiment no model is good (samples 1,2,4). In all cases **OCSVM** does not correctly predict the direction of for reusability for the model that yields best results. (sample 3)

#### Model Choice

- **std data, std in performance** - except for MMD & OCSVM pairs ((pi2, pi3) sample 1,2,3  and (pi3, pi5) sample 1) where std for both nodes is small, the rest of the pairs std is high and model choice greatly affects the OCSVM diretionality predictability. This is not true for sample 4 as depending on the pairing std is very high for node and there seems to be no pattern and before it was true that MMD, OCSVM pairs has low std this is not true in this case.

    ***In all cases except for pair (pi2, pi3) and pair (pi3, pi5) only for node pi5 in sample 3 choosing the model with lowest performance yields the lowest discrepancy and best results. Only difference for sample 4 is that for MMD and OCSVM pairs it depends on the node***

- **non-std data, std in performance** - std for models varies and it is higher than 0.1 in most cases.

    ***Sample 1 & 4***:

    ***For nodes pi2, pi4 and pi5 the non-linear model has the best baseline model performance and the lowest discrepancy in all pairrings, while for pi3 in terms of the discrepancy the opposite is true.***
    
    ***Sample 2***:
    
    ***For node pi2 the non-linear model has the best baseline model performance and the lowest discrepancy in all pairrings, while for the rest of nodes (pi3, pi4, pi5) in terms of the discrepancy the opposite is true.***
    
    ***Sample 3***:

    ***For node pi2 the non-linear model has the best baseline model performance and the lowest discrepancy in all pairrings, while for pi3 in terms of the discrepancy the opposite is true. For nodes pi4 and pi5 is not as clear and it depends on the pairring***
    
### Experiment 3

- **Node baseline models**: pi2, pi4 (above 0.9 R2 - for both std and non-std data), std: pi5 (sample 1: above 0.5, samples 2,3: just above 0.6, sample 4: just below 0.6) non-std: pi5 (just above 0.4 (sample 1 &4), 0.5 (sample 2 &3)) and std: pi3 (samples 1,2: below 0.2, sample 3: just above 0.3, sample 4: just below 0.3), non-std: pi3 (just above 0.3)

- Pairs where both models have good baseline scores are good models for reusability on both sides (only when models are linear for non-std data). Note on std data: **OCSVM, MMD pairs** good for reusability on both sides (OCSVM higher than 0.73, 0.72, 0.7 in both sides). Note on non-std data: ***MMD*** pairs unstable 

- For std data: OCSVM correctly predicts the direction of for reusability for the model that yields best results, however the opposite direction can have a better model if the correct model is not chosen. ***except for pair pi2 and pi5 where actually pi2 has higher r2 - discrepancy perfomance than the native model for pi5. Also for pair pi4 and pi5 the model linear model of pi4 when used with pi5 data yeilds very similar score to the native model*** (sample 3) For non-std data: **OCSVM** does not correctly predict the directionality. Note: sample 2 has no OCSVM pairs

#### Model Choice

- For all pairs (forward, backward) non-linear models have higher **baseline scores**. For std data also true that on averge non-linear models have lower discrepnacy (samples 1,2,3)

**std data**

***Sample 1 & 2***:

- **std**: with whatever node pi3 is paired with the std is small, pi5 has consistenly a high std and it's std is always higher than it's counterpart, for nodes pi2 and pi4 it depends on the node paired with however, when paired with a node with a good model std remains low. (opposite statements for nodes pi2, pi3 and pi4 in sample 2)

    ***Sample 1: Therefore for node pi3 the kernel choice does not matter too much as it seems it does not affect the discrepancy and therefore model choice should be based on native performance on node. For node pi5 the non-linear kernel performs best with all pair combinations. While for nodes pi2 and pi4 it solely depends on the pairing***

    ***Sample 2: Therefore for nodes pi2 and pi4 the kernel choice does not matter too much as it seems it does not affect the discrepancy and therefore model choice should be based on native performance on node. For nodes pi3 and pi5 the non-linear kernel performs best with all pair combinations.***
    
***Sample 3&4***: 
- **std**: varies dependsing on the pair, pair (pi2, pi4) has the lowest std and nodes pi2 and pi4 have low std when paired with pi3. Node pi5 has lower std compared to the node its paired with (sample 3 opposite for sample 4)

    ***For nodes pi2 and pi4 the linear model has similar performance to the non-linear one but lower discrepancy especially with models where baseline performance is npt good, while for node pi3, pi5 the non-linear model always has a higher score and lower discrepancy.***
    
**non-std data**
    
- **std**: std varies depending on the pairing. It is not easy to infer the model to choose per node, for example for node pi4 the linear model is better when paired with node pi3 but the non-linear model is better when paired with pi2.
